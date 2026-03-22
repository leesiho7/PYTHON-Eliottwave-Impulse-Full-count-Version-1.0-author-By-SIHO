import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import json
import logging
import io

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI 클라이언트 설정
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class MarketData(BaseModel):
    symbol: str
    current_price: float
    df_json: str
    question: str = "현재 시장 상황을 분석해줘"

def analyze_with_gpt(symbol, current_price, rsi_info, df_context, user_query):
    prompt = f"""
    너는 암호화폐 기술 분석 대가 'Short Specialist'이자 'Dober AI'야. 
    사용자의 질문 "{user_query}"에 대해 아래의 **3단계 리포트 형식**으로 답변해줘.

    **[Step 1. 파동 및 지표 현황]**
    - 현재 엘리어트 파동 위치와 RSI 상태 요약.

    **[Step 2. Short Specialist 핵심 전략 검토]**
    - '3파 과매수(RSI 70이상) -> 4파 조정 -> 5파 신고가 및 RSI 하락 다이버전스' 로직 적용 여부 체크.

    **[Step 3. 향후 대응 시나리오]**
    - STAY/ENTER 결정 및 구체적인 타겟 가격 조언.

    [시장 데이터] 종목: {symbol}, 가격: {current_price}, 현재 RSI: {rsi_info['current_rsi']:.2f}
    [차트 데이터 (최근 15봉)]
    {df_context}

    **반드시 JSON으로만 응답해.**
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "너는 질문의 의도를 파악하여 기술적 로직과 시나리오를 결합해 답변하는 퀀트 전문가다. 반드시 JSON으로 응답하라."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" }
        )
        res_content = response.choices[0].message.content
        return json.loads(res_content)
    except Exception as e:
        logger.error(f"GPT 처리 에러: {e}")
        return {
            "decision": "STAY",
            "reason": "AI 분석 노드가 연산 중입니다. 잠시 후 다시 시도해 주세요.",
            "entry_price": current_price, "stop_loss": 0, "take_profit": 0
        }

@app.post("/api/v1/analyze-short")
async def run_strategy(data: MarketData):
    try:
        # 1) 데이터 로드 및 방어
        if not data.df_json or len(data.df_json) < 20:
            return {"decision": "STAY", "reason": "실시간 차트 데이터 수집량이 부족합니다. 캔들이 더 쌓일 때까지 기다려 주세요."}

        try:
            df = pd.read_json(io.StringIO(data.df_json))
        except:
            df = pd.read_json(data.df_json)

        # 컬럼명 전처리 (모두 소문자로)
        df.columns = [str(c).lower() for c in df.columns]

        # 종가(close) 컬럼 찾기 방어 로직
        close_candidates = ['close', 'c', 'last', 'price', '4'] # Bybit 인덱스 4 대응
        close_col = next((c for c in df.columns if c in close_candidates), df.columns[-1])

        # 2) RSI 계산 및 결측치 방어
        delta = df[close_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().replace(0, 0.001)
        
        df['rsi'] = 100 - (100 / (1 + (gain / loss)))
        df['rsi'] = df['rsi'].fillna(50) # 계산 안된 구간 기본값 채우기

        current_rsi = float(df['rsi'].iloc[-1])
        
        # 3) 특화 필터링
        if current_rsi < 35 and "롱" not in data.question:
            return {
                "decision": "STAY",
                "reason": f"현재 RSI가 {current_rsi:.2f}로 과매도 상태입니다. 지금 숏 진입은 반등 위험이 크므로 5파 반등 후 고점 다이버전스를 기다리세요."
            }

        # 4) 컨텍스트 준비 (사용 가능한 컬럼만 요약)
        context_cols = [c for c in ['high', close_col, 'rsi'] if c in df.columns]
        recent_context = df.tail(15)[context_cols].to_string(index=False)
        rsi_info = {'current_rsi': current_rsi}

        # 5) GPT 실행
        result = analyze_with_gpt(data.symbol, data.current_price, rsi_info, recent_context, data.question)
        
        # 자바가 기다리는 'reason' 키값 보장
        if "reason" not in result:
            result["reason"] = result.get("analysis", "분석이 완료되었습니다.")
            
        return result

    except Exception as e:
        logger.error(f"전체 에러: {e}")
        return {"decision": "STAY", "reason": f"시스템 체크: 데이터 형식을 확인 중입니다. ({str(e)[:50]})"}

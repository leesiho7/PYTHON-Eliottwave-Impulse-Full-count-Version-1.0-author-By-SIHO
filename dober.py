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
    # GPT에게 형식을 강제하는 프롬프트
    prompt = f"""
    너는 최고의 암호화폐 분석가이자 'Short Specialist'인 'Dober AI'야.
    사용자의 질문 "{user_query}"에 대해 아래 3단계 형식을 엄격히 지켜 답변해줘.

    **[Step 1. 파동 및 지표 현황]**
    - 현재 엘리어트 파동 위치와 RSI 상태 요약.
    
    **[Step 2. 전략 체크리스트]**
    - '3파 과매수 -> 4파 조정 -> 5파 하락 다이버전스' 로직 충족 여부 체크.
    
    **[Step 3. 향후 대응 시나리오]**
    - STAY/ENTER 결정 및 구체적인 타겟 가격 조언.

    [데이터] 종목:{symbol}, 가격:{current_price}, RSI:{rsi_info['current_rsi']:.2f}
    [차트흐름 (최근 15봉)]
    {df_context}

    **반드시 답변 내용을 'reason'이라는 키에 담아 JSON으로 응답해.**
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "너는 무조건 {'decision': '...', 'reason': '...'} 형식으로만 응답하는 전문 퀀트 분석가다."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" }
        )
        
        res_content = response.choices[0].message.content
        logger.info(f"🤖 GPT 원본 응답: {res_content}")
        
        res_json = json.loads(res_content)
        
        # [핵심 수정] 어떤 키값으로 오든 'reason'을 찾아내는 로직
        final_reason = ""
        if "reason" in res_json:
            final_reason = res_json["reason"]
        elif "analysis" in res_json:
            final_reason = res_json["analysis"]
        elif "Step 1" in res_content: # 키가 깨졌지만 내용은 있을 때
            final_reason = res_content 
        else:
            final_reason = "분석 리포트 생성 중 응답 규격 오류가 발생했습니다. 다시 시도해 주세요."

        return {
            "decision": res_json.get("decision", "STAY"),
            "reason": final_reason,
            "entry_price": current_price
        }
        
    except Exception as e:
        logger.error(f"❌ GPT 처리 에러: {e}")
        return {"decision": "STAY", "reason": f"AI 노드 연산 중 오류가 발생했습니다: {str(e)[:50]}"}

@app.post("/api/v1/analyze-short")
async def run_strategy(data: MarketData):
    try:
        # 1) 데이터 로딩 방어막
        if not data.df_json or len(data.df_json) < 20:
            return {"decision": "STAY", "reason": "실시간 차트 데이터 수집량이 부족합니다. 캔들이 더 쌓일 때까지 기다려 주세요."}

        try:
            df = pd.read_json(io.StringIO(data.df_json))
        except:
            df = pd.read_json(data.df_json)

        df.columns = [str(c).lower() for c in df.columns]
        
        # 종가(close) 컬럼 자동 매칭
        close_names = ['close', 'c', 'last', '4']
        close_col = next((c for c in df.columns if c in close_names), df.columns[-1])

        # 2) RSI 계산 (안전 모드)
        delta = df[close_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().replace(0, 0.001)
        
        df['rsi'] = 100 - (100 / (1 + (gain / loss)))
        df['rsi'] = df['rsi'].fillna(50)

        current_rsi = float(df['rsi'].iloc[-1])
        
        # 3) 숏 전략 예외 필터
        if current_rsi < 35 and "롱" not in data.question:
            return {
                "decision": "STAY",
                "reason": f"현재 RSI({current_rsi:.2f})가 과매도권입니다. 지금 숏 진입은 반등 위험이 크므로, 5파동 반등 후 고점 다이버전스를 기다려야 합니다."
            }

        # 4) 컨텍스트 준비 (최근 15봉)
        recent_context = df.tail(15)[['high', close_col, 'rsi']].to_string(index=False)
        rsi_info = {'current_rsi': current_rsi}

        # 5) 마스터 분석 실행
        return analyze_with_gpt(data.symbol, data.current_price, rsi_info, recent_context, data.question)

    except Exception as e:
        logger.error(f"❌ 전체 프로세스 에러: {e}")
        return {"decision": "STAY", "reason": f"데이터 분석 실패: {str(e)[:50]}"}

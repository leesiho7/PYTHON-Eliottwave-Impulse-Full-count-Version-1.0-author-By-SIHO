import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import json
import logging

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Railway Variables에 등록한 OPENAI_API_KEY 사용
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class MarketData(BaseModel):
    symbol: str
    current_price: float
    df_json: str 

def calculate_rsi(df, period=14):
    try:
        # 컬럼명 유연하게 찾기 (대소문자 및 숫자형 방어)
        target_col = None
        for col in df.columns:
            if str(col).lower() == 'close':
                target_col = col
                break
        
        if target_col is None:
            return pd.Series([0] * len(df))

        delta = df[target_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    except Exception as e:
        logger.error(f"RSI 계산 오류: {e}")
        return pd.Series([0] * len(df))

def analyze_with_gpt(symbol, current_price, rsi_info, df_context):
    # 시나리오 분석(4번) 기능이 강화된 프롬프트
    prompt = f"""
    너는 엘리어트 파동과 RSI 다이버전스 분석의 대가 'Short Specialist'야.
    [데이터] 종목: {symbol}, 현재가: {current_price}, 현재 RSI: {rsi_info['current_rsi']:.2f}
    [최근 흐름]
    {df_context}

    [미션] 
    1. 현재가 엘리어트 5파 종료 구간인지, RSI 하락 다이버전스가 발생하는지 정밀 분석해.
    2. 진입 조건이 아닐 경우(STAY), 단순히 거절하지 말고 '향후 어떤 가격대나 RSI 수치'가 도달했을 때 숏 진입이 유리할지 시나리오를 제시해.
    3. 진입 조건일 경우(ENTER), 논리적인 손절가(SL)와 익절가(TP)를 숫자로만 제시해.

    반드시 JSON으로만 답변해: 
    {{ 
        "decision": "ENTER" 또는 "STAY", 
        "reason": "전문가 관점의 분석과 향후 가격 시나리오를 포함한 한줄 요약", 
        "entry_price": {current_price}, 
        "stop_loss": 숫자로계산값, 
        "take_profit": 숫자로계산값 
    }}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "너는 금융 데이터만 보고 타점을 잡는 냉철한 퀀트 트레이더다. JSON으로만 응답하라."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"GPT 호출 에러: {e}")
        return {"decision": "STAY", "reason": "AI 분석 노드 통신 지연", "entry_price": current_price, "stop_loss": 0, "take_profit": 0}

@app.post("/api/v1/analyze-short")
async def run_strategy(data: MarketData):
    try:
        # 1) JSON 데이터 로드
        try:
            df = pd.read_json(data.df_json)
        except:
            import io
            df = pd.read_json(io.StringIO(data.df_json))

        # 2) 모든 컬럼명을 문자열로 변환 후 소문자화 (숫자형 컬럼 에러 방지)
        df.columns = [str(c).lower() for c in df.columns]

        # 3) 필수 필드 체크
        if 'close' not in df.columns:
            return {"decision": "STAY", "reason": f"데이터 필드명 불일치 (받은 항목: {list(df.columns)})"}

        # 4) 기술적 지표 계산
        df['rsi'] = calculate_rsi(df)
        current_rsi = float(df['rsi'].iloc[-1])
        
        # 5) 전략 필터링 및 컨텍스트 요약
        # RSI가 너무 낮을 때(과매도)는 GPT를 부르지 않고 즉시 시나리오 답변 (비용 절약)
        if current_rsi < 40:
            return {
                "decision": "STAY", 
                "reason": f"현재 RSI({current_rsi:.2f})가 과매도권에 있어 숏 진입은 위험합니다. 반등을 기다린 후 70 이상의 과매수 구간에서 하락 다이버전스를 확인하세요."
            }

        # 최근 15개 캔들 데이터를 GPT에게 전달
        recent_context = df.tail(15)[['high', 'close', 'rsi']].to_string(index=False)
        rsi_info = {'current_rsi': current_rsi, 'max_rsi_3p': float(df['rsi'].max())}

        # 6) GPT 전문가 분석 실행
        return analyze_with_gpt(data.symbol, data.current_price, rsi_info, recent_context)

    except Exception as e:
        logger.error(f"시스템 체크 에러: {e}")
        return {"decision": "STAY", "reason": f"System Check: {str(e)}"}
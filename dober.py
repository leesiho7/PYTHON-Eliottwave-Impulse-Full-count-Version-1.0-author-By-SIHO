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

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class MarketData(BaseModel):
    symbol: str
    current_price: float
    df_json: str
    question: str = "현재 시장 상황을 분석해줘"

def analyze_with_gpt(symbol, current_price, rsi_info, df_context, user_query):
    # 다양한 질문(숏, 롱, 파동 등)을 모두 소화할 수 있는 마스터 프롬프트
    prompt = f"""
    너는 최고의 암호화폐 기술 분석가이자 엘리어트 파동/RSI 전문가 'Dober AI'야.
    
    [사용자 질문]: "{user_query}"
    [시장 데이터]: 종목 {symbol}, 현재가 {current_price}, RSI {rsi_info['current_rsi']:.2f}
    [차트 흐름 (최근 15캔들)]:
    {df_context}

    [미션 가이드]
    1. 사용자의 질문 의도를 파악해 (숏/롱 포지션 문의, 파동 카운팅 요청, 단순 시황 등).
    2. 질문이 '롱'에 관한 것이라도, 너의 메인 전략인 '엘리어트 5파 종료 후 숏 타점' 관점을 유지하며 균형 있게 답변해.
    3. '파동 분석해봐'라는 질문에는 현재가 몇 파동 진행 중인지(1~5파 또는 ABC) 너의 추정치를 반드시 언급해.
    4. STAY(관망)라면, 사용자가 진입을 고려해볼 만한 '기준 가격'이나 'RSI 수치'를 시나리오로 제시해.

    반드시 아래 JSON 형식으로만 응답해:
    {{
        "decision": "ENTER/STAY/EXIT",
        "reason": "사용자 질문에 대한 직접적인 답변 + 파동/RSI 근거 + 향후 시나리오 요약",
        "entry_price": {current_price},
        "stop_loss": 숫자로계산값,
        "take_profit": 숫자로계산값
    }}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "너는 질문의 의도를 정확히 파악하여 기술적 근거와 함께 답변하는 트레이딩 전문가다."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"GPT Error: {e}")
        return {"decision": "STAY", "reason": "AI 분석 엔진 일시 지연 중..", "entry_price": current_price, "stop_loss": 0, "take_profit": 0}

@app.post("/api/v1/analyze-short")
async def run_strategy(data: MarketData):
    try:
        # 1) 데이터 로드 및 전처리
        df = pd.read_json(data.df_json)
        df.columns = [str(c).lower() for c in df.columns]
        
        # 2) RSI 계산 (14 period)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['rsi'] = 100 - (100 / (1 + (gain / loss)))
        
        current_rsi = float(df['rsi'].iloc[-1])
        
        # 3) 예외 케이스: RSI가 극단적으로 낮을 때 (숏 위험 구간)
        # 질문에 '롱'이나 '반등'이 포함된 경우 GPT에게 판단을 맡기기 위해 필터를 유연하게 적용
        if current_rsi < 30 and "롱" not in data.question:
            return {
                "decision": "STAY",
                "reason": f"현재 RSI가 {current_rsi:.2f}로 과매도권입니다. 지금 숏 진입은 매우 위험하며, 기술적 반등을 기다린 후 고점에서 다이버전스를 확인하는 것이 안전합니다."
            }

        # 4) GPT 분석 컨텍스트 준비
        recent_context = df.tail(15)[['high', 'close', 'rsi']].to_string(index=False)
        rsi_info = {'current_rsi': current_rsi}

        # 5) GPT 호출
        return analyze_with_gpt(data.symbol, data.current_price, rsi_info, recent_context, data.question)

    except Exception as e:
        logger.error(f"Global Error: {e}")
        return {"decision": "STAY", "reason": f"System Check: {str(e)}"}
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

# Railway 환경변수에서 API 키 로드
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class MarketData(BaseModel):
    symbol: str
    current_price: float
    df_json: str
    question: str = "현재 시장 상황을 분석해줘"

def analyze_with_gpt(symbol, current_price, rsi_info, df_context, user_query):
    # [핵심 로직 고정 + 풍성한 대화]를 위한 마스터 프롬프트
    prompt = f"""
    너는 암호화폐 기술 분석 대가 'Short Specialist'이자 'Dober AI'야. 
    사용자의 질문 "{user_query}"에 대해 아래의 **3단계 리포트 형식**으로 답변해줘.

    **[Step 1. 파동 및 지표 현황]**
    - 현재가 엘리어트 파동 관점에서 몇 파동(예: 4파 조정 중)인지, RSI 상태는 어떤지 요약해.

    **[Step 2. Short Specialist 핵심 전략 검토]**
    - 나의 핵심 숏 전략인 **'3파 과매수(RSI 70이상) -> 4파 조정 -> 5파 신고가 및 RSI 하락 다이버전스'** 로직을 적용해.
    - 현재 이 조건 중 무엇이 충족되었고, 무엇이 미달인지 '체크리스트' 형태로 냉정하게 짚어줘.

    **[Step 3. 향후 대응 시나리오]**
    - 질문에 대한 직접적인 답변과 함께, STAY라면 "어느 가격대까지 기다려야 하는지", ENTER라면 "손절/익절가"를 전략적으로 조언해줘.

    [시장 데이터] 종목: {symbol}, 가격: {current_price}, 현재 RSI: {rsi_info['current_rsi']:.2f}
    [차트 데이터 (최근 15봉)]
    {df_context}

    **주의: 결론은 냉철한 로직에 근거하되, 말투는 파트너에게 신뢰를 주는 전문적인 톤을 유지해.**
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "너는 질문의 의도를 파악하여 기술적 로직과 시나리오를 결합해 답변하는 퀀트 전문가다. 반드시 JSON으로만 응답하라."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"GPT Error: {e}")
        return {
            "decision": "STAY", 
            "reason": "AI 분석 엔진의 연산이 지연되고 있습니다. 잠시 후 다시 시도해주세요.",
            "entry_price": current_price, "stop_loss": 0, "take_profit": 0
        }

@app.post("/api/v1/analyze-short")
async def run_strategy(data: MarketData):
    try:
        # 1) 데이터 전처리
        df = pd.read_json(data.df_json)
        df.columns = [str(c).lower() for c in df.columns]
        
        # 2) RSI 계산
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['rsi'] = 100 - (100 / (1 + (gain / loss)))
        
        current_rsi = float(df['rsi'].iloc[-1])
        
        # 3) 예외 처리: RSI가 너무 낮을 때 (숏 전략가 관점에서 강력한 반등 경고)
        # 단, 사용자가 '롱'에 대해 물어봤다면 GPT의 유연한 판단을 위해 필터 통과
        if current_rsi < 35 and "롱" not in data.question:
            return {
                "decision": "STAY",
                "reason": f"현재 RSI가 {current_rsi:.2f}로 과매도권에 진입했습니다. 나의 'Short Specialist' 로직상 지금 숏은 매우 위험하며, 5파동 반등이 나온 후 고점에서 다이버전스가 형성될 때까지 관망해야 합니다."
            }

        # 4) 분석 컨텍스트 준비 (GPT에게 전달할 핵심 데이터 요약)
        recent_context = df.tail(15)[['high', 'close', 'rsi']].to_string(index=False)
        rsi_info = {'current_rsi': current_rsi}

        # 5) 마스터 분석 실행
        return analyze_with_gpt(data.symbol, data.current_price, rsi_info, recent_context, data.question)

    except Exception as e:
        logger.error(f"Global Error: {e}")
        return {"decision": "STAY", "reason": f"시스템 체크 중 오류 발생: {str(e)}"}
import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import json
import logging

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class MarketData(BaseModel):
    symbol: str
    current_price: float
    df_json: str
    question: str  # [추가] 자바에서 넘어오는 사용자 질문

def analyze_with_gpt(symbol, current_price, rsi_info, df_context, user_query):
    # 프롬프트에 사용자의 구체적인 질문(user_query)을 포함시킵니다.
    prompt = f"""
    너는 엘리어트 파동과 RSI 분석 대가 'Short Specialist'야.
    
    [사용자 질문]: "{user_query}"
    [시장 데이터]: 종목 {symbol}, 현재가 {current_price}, RSI {rsi_info['current_rsi']:.2f}
    [차트 흐름]:
    {df_context}

    [미션]
    1. 사용자의 질문에 직접적으로 답하면서, 차트 근거(파동, RSI)를 제시해.
    2. 현재가 숏 진입 적기인지(ENTER) 아니면 더 기다려야 하는지(STAY) 결정해.
    3. STAY라면, "어떤 가격대까지 반등했을 때" 다시 분석할지 시나리오를 구체적으로 언급해.

    반드시 JSON으로만 응답해:
    {{
        "decision": "ENTER/STAY",
        "reason": "사용자 질문에 대한 답변 + 차트 분석 + 향후 시나리오를 포함한 친절한 설명",
        "entry_price": {current_price},
        "stop_loss": 숫자로계산,
        "take_profit": 숫자로계산
    }}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "너는 차트 데이터와 사용자 질문을 결합해 전략을 짜주는 퀀트 전문가다."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except:
        return {"decision": "STAY", "reason": "AI 분석 노드 점검 중입니다."}

@app.post("/api/v1/analyze-short")
async def run_strategy(data: MarketData):
    try:
        df = pd.read_json(data.df_json)
        df.columns = [str(c).lower() for c in df.columns]
        
        # 지표 계산
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + (gain / loss)))
        
        current_rsi = float(df['rsi'].iloc[-1])
        recent_context = df.tail(15)[['high', 'close', 'rsi']].to_string(index=False)
        rsi_info = {'current_rsi': current_rsi, 'max_rsi_3p': float(df['rsi'].max())}

        # GPT에게 질문과 데이터를 함께 전달
        return analyze_with_gpt(data.symbol, data.current_price, rsi_info, recent_context, data.question)
    except Exception as e:
        return {"decision": "STAY", "reason": f"System Check: {str(e)}"}
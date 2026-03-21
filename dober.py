import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import json

# 1. 설정 및 모델 초기화
app = FastAPI()

# Railway Variables에 등록한 OPENAI_API_KEY를 자동으로 가져옵니다.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 데이터 구조 정의 (자바에서 넘겨줄 데이터 포맷)
class MarketData(BaseModel):
    symbol: str
    current_price: float
    df_json: str  # 자바에서 전달한 캔들 데이터(JSON String)

# 2. 도메인 로직: RSI 계산
def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# 3. 에이전트 핵심 분석 함수 (GPT-4 버전)
def analyze_with_gpt(symbol, current_price, rsi_info, df_context):
    prompt = f"""
    너는 엘리어트 파동과 RSI 다이버전스 분석의 대가 'Short Specialist'야.
    [데이터]
    - 종목: {symbol} / 현재가: {current_price}
    - 현재 RSI: {rsi_info['current_rsi']:.2f} / 3파 최고 RSI: {rsi_info['max_rsi_3p']:.2f}
    - 최근 흐름 (OHLCV & RSI): 
    {df_context}

    [미션]
    현재가 5파 종료 구간인지, 그리고 RSI 하락 다이버전스가 뚜렷한지 분석해.
    반드시 다음 JSON 형식으로만 응답해. 텍스트 설명은 절대로 포함하지 마.
    {{
        "decision": "ENTER" 또는 "STAY",
        "reason": "분석 근거 한줄",
        "entry_price": {current_price},
        "stop_loss": 손절가(숫자),
        "take_profit": 익절가(숫자)
    }}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",  # 혹은 "gpt-4o"
            messages=[
                {"role": "system", "content": "너는 금융 분석 전문가이며 JSON으로만 응답하는 봇이다."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" } # JSON 출력 강제
        )
        
        res_text = response.choices[0].message.content
        return json.loads(res_text)
    except Exception as e:
        return {"decision": "STAY", "reason": f"AI Error: {str(e)}", "entry_price": current_price, "stop_loss": 0, "take_profit": 0}

# 4. 자바 컨트롤러가 호출할 API 엔드포인트
@app.post("/api/v1/analyze-short")
async def run_strategy(data: MarketData):
    try:
        # 1) JSON 데이터를 DataFrame으로 복구
        df = pd.read_json(data.df_json)
        df['rsi'] = calculate_rsi(df)
        
        current_rsi = df['rsi'].iloc[-1]
        max_rsi_3p = df['rsi'].tail(50).max()
        
        # 2) 전략 실행 기본 조건 (RSI 60 미만이면 AI 호출 안함 - 리소스 절약)
        if current_rsi < 60:
            return {"decision": "STAY", "reason": "RSI condition not met (Under 60)"}

        # 3) 최근 20개 캔들 요약 (GPT에게 전달할 컨텍스트)
        recent_context = df.tail(20)[['high', 'close', 'rsi']].to_string()
        rsi_info = {'current_rsi': current_rsi, 'max_rsi_3p': max_rsi_3p}

        # 4) GPT 에이전트 분석 호출
        final_analysis = analyze_with_gpt(data.symbol, data.current_price, rsi_info, recent_context)
        
        return final_analysis

    except Exception as e:
        return {"decision": "STAY", "reason": f"Server Error: {str(e)}"}

# uvicorn 실행 시: uvicorn dober:app --host 0.0.0.0 --port 8000

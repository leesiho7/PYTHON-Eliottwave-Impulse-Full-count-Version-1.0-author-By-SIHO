import pandas as pd
import google.generativeai as genai
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import json

# 1. 설정 및 모델 초기화
app = FastAPI()
genai.configure(api_key="시호의_GEMINI_API_KEY")
model = genai.GenerativeModel('gemini-pro')

# 데이터 구조 정의 (자바에서 넘겨줄 데이터)
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

# 3. 에이전트 핵심 분석 함수
def analyze_with_gemini(symbol, current_price, rsi_info, df_context):
    prompt = f"""
    너는 엘리어트 파동과 RSI 다이버전스 분석의 대가 'Short Specialist'야.
    [데이터]
    - 종목: {symbol} / 현재가: {current_price}
    - 현재 RSI: {rsi_info['current_rsi']:.2f} / 3파 최고 RSI: {rsi_info['max_rsi_3p']:.2f}
    - 최근 흐름: {df_context}

    [미션]
    하락 다이버전스와 5파 종료 여부를 분석하여 다음 JSON 형식으로만 답해. 텍스트 설명은 제외해.
    {{
        "decision": "ENTER" 또는 "STAY",
        "reason": "분석 근거 한줄",
        "entry_price": {current_price},
        "stop_loss": 손절가계산값,
        "take_profit": 익절가계산값
    }}
    """
    try:
        response = model.generate_content(prompt)
        # JSON 부분만 추출 (가끔 AI가 설명을 붙일 때를 대비)
        res_text = response.text.strip()
        return json.loads(res_text[res_text.find('{'):res_text.rfind('}')+1])
    except:
        return {"decision": "STAY", "reason": "AI Error"}

# 4. 자바 컨트롤러가 호출할 API 엔드포인트
@app.post("/api/v1/analyze-short")
async def run_strategy(data: MarketData):
    # JSON 데이터를 DataFrame으로 복구
    df = pd.read_json(data.df_json)
    df['rsi'] = calculate_rsi(df)
    
    current_rsi = df['rsi'].iloc[-1]
    max_rsi_3p = df['rsi'].tail(50).max()
    
    # 전략 실행 조건 (필터링)
    if current_rsi < 60:
        return {"decision": "STAY", "reason": "RSI condition not met"}

    # 최근 20개 캔들 요약
    recent_context = df.tail(20)[['high', 'close', 'rsi']].to_string()
    rsi_info = {'current_rsi': current_rsi, 'max_rsi_3p': max_rsi_3p}

    # Gemini 에이전트 호출
    final_analysis = analyze_with_gemini(data.symbol, data.current_price, rsi_info, recent_context)
    
    return final_analysis

# 실행: uvicorn main:app --reload
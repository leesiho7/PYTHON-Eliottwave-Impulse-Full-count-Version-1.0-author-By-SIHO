import os
import time
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

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class MarketData(BaseModel):
    symbol: str
    current_price: float
    df_json: str
    question: str = "현재 시장 상황을 분석해줘"

def analyze_with_gpt(symbol, current_price, rsi_info, df_summary, user_query, macro_info):
    # [수정] df_context를 df_summary로 통일하고, 좌표 추출 지시를 더 명확히 함
    prompt = f"""
    너는 HTF(거시 추세)를 읽고 '3-4-5파 하락 다이버전스'를 포착하는 'Short Specialist' Dober AI야.
    
    [핵심 미션]
    1. 사용자의 질문 "{user_query}"에 대해 기술적으로 분석하라.
    2. 제공된 [차트 데이터]의 timestamp를 사용하여 엘리어트 파동(1,2,3,4,5파)의 정확한 좌표를 'visualData'에 담아라.

    [차트 데이터 (과거 요약 + 최근 정밀)]
    {df_summary}

    [현재 상황]
    - 종목: {symbol}, 현재가: {current_price}, RSI: {rsi_info['current_rsi']:.2f}
    - 거시적 관점: {macro_info}

    [출력 규칙]
    반드시 JSON으로만 응답하며, 'points'의 timestamp는 반드시 위 [차트 데이터]에 존재하는 실제 timestamp 값을 그대로 사용할 것. 예시 값을 절대 사용하지 말 것.
    {{
        "decision": "ENTER/STAY",
        "reason": "분석 리포트 내용",
        "visualData": {{
            "points": [
                {{"label": "1", "type": "low", "timestamp": <차트_데이터의_실제_timestamp>, "price": <실제_가격>}},
                {{"label": "3", "type": "high", "timestamp": <차트_데이터의_실제_timestamp>, "price": <실제_가격>}},
                {{"label": "4", "type": "low", "timestamp": <차트_데이터의_실제_timestamp>, "price": <실제_가격>}}
            ]
        }}
    }}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "너는 엘리어트 파동 좌표 추출 전문가다. 데이터의 timestamp와 price를 정확히 매칭하라."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"GPT Error: {e}")
        return {"decision": "STAY", "reason": "AI 연산 지연 중.."}

@app.post("/api/v1/analyze-short")
async def run_strategy(data: MarketData):
    try:
        df = pd.read_json(io.StringIO(data.df_json))

        # [FIX 3] 숫자 컬럼명("0","1",...) → 명시적 컬럼명으로 변환
        if df.columns[0] in [0, '0']:
            col_names = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']
            df.columns = col_names[:len(df.columns)]
        else:
            df.columns = [str(c).lower() for c in df.columns]

        close_col = next((c for c in df.columns if c in ['close', 'c', '4']), df.columns[-1])
        time_col = next((c for c in df.columns if 'time' in c), df.columns[0])

        # [FIX 1] 타임스탬프 현재 시점(2026)으로 동기화
        current_time_ms = int(time.time() * 1000)
        last_ts = int(df[time_col].iloc[-1])
        time_offset = current_time_ms - last_ts
        if abs(time_offset) > 86400000:  # 1일 이상 차이 날 때만 보정
            df[time_col] = df[time_col].astype(int) + time_offset
            logger.info(f"타임스탬프 보정 적용: offset={time_offset}ms")

        # 1) RSI 계산
        delta = df[close_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().replace(0, 0.001)
        df['rsi'] = 100 - (100 / (1 + (gain / loss)))
        df['rsi'] = df['rsi'].fillna(50)

        # 2) [보완] 데이터 요약 및 정렬
        df_recent = df.tail(20)
        df_old = df.iloc[:-20].tail(80).iloc[::5]
        # 합친 후 시간순으로 정렬하여 GPT의 혼란 방지
        df_combined = pd.concat([df_old, df_recent]).sort_values(by=time_col)
        
        df_summary = df_combined[[time_col, 'high', 'low', close_col, 'rsi']].to_string(index=False)

        # 3) HTF 분석
        current_rsi = float(df['rsi'].iloc[-1])
        macro_info = "거시 상승 추세 (강력한 지지)" if df[close_col].iloc[-1] > df[close_col].iloc[-50] else "박스권 혹은 하락 추세"

        return analyze_with_gpt(data.symbol, data.current_price, {'current_rsi': current_rsi}, df_summary, data.question, macro_info)

    except Exception as e:
        logger.error(f"Global Error: {e}")
        return {"decision": "STAY", "reason": f"시스템 체크 중 오류: {str(e)[:50]}"}

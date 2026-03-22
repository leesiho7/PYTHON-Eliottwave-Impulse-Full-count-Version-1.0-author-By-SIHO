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

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class MarketData(BaseModel):
    symbol: str
    current_price: float
    df_json: str
    question: str = "현재 시장 상황을 분석해줘"

def analyze_with_gpt(symbol, current_price, rsi_info, df_summary, user_query, macro_info):
    prompt = f"""
    너는 HTF(거시 추세)를 읽고 '3-4-5파 하락 다이버전스'를 포착하는 'Short Specialist' Dober AI야.
    아래 가이드라인에 따라 사용자의 질문 "{user_query}"에 답변해줘.
    
    **[핵심 분석 로직: 숏 타점 포착]**
    1. **3파 확인**: 최근 구간 중 RSI가 70 이상(과매수)으로 치솟으며 강력한 상승을 보인 지점.
    2. **4파 조정**: 3파 이후 RSI가 식으면서 이전 RSI 고점을 하향 돌파한 지점.
    3. 아래 제공된 [차트 요약 데이터]에서 엘리어트 파동의 변곡점(1,2,3,4파)을 찾아 'visual_data' 좌표를 생성해.
    
    **[거시적 관점(HTF)]**: {macro_info}
    - 만약 '히든 상승 다이버전스'가 감지되었다면 숏 진입은 극도로 보수적으로 판단할 것.
    **[현재 데이터]** 종목: {symbol}, 가격: {current_price}, RSI: {rsi_info['current_rsi']:.2f}
    **[차트 흐름(샘플링)]**
    {df_context}

    
    [데이터 정보]
    - 종목: {symbol}, 현재가: {current_price}, RSI: {rsi_info['current_rsi']:.2f}
    - 거시 상황: {macro_info}
    - 차트 요약(최근 100봉 중 주요 지점):
    {df_summary}

    [출력 규칙]
    반드시 JSON으로 응답하며, 'points'의 timestamp는 반드시 제공된 데이터의 값을 그대로 사용해.
    {{
        "decision": "ENTER/STAY",
        "reason": "분석 리포트",
        "visual_data": {{
            "points": [
                {{"label": "3", "type": "high", "timestamp": 1711080000000, "price": 70885.5}},
                {{"label": "4", "type": "low", "timestamp": 1711123200000, "price": 68723.2}}
            ]
        }}
    }}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "너는 엘리어트 파동 좌표 추출 전문가다. 타임스탬프 오차를 허용하지 않는다."},
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
        df.columns = [str(c).lower() for c in df.columns]
        close_col = next((c for c in df.columns if c in ['close', 'c', '4']), df.columns[-1])
        time_col = next((c for c in df.columns if 'time' in c), df.columns[0])

        # 1) RSI 및 지표 계산 (정밀도 유지)
        delta = df[close_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().replace(0, 0.001)
        df['rsi'] = 100 - (100 / (1 + (gain / loss)))
        df['rsi'] = df['rsi'].fillna(50)

        # 2) [핵심 수정] GPT에게 보낼 '정밀 요약본' 생성
        # 단순히 건너뛰는 게 아니라, 최근 20봉은 정밀하게 + 이전 80봉은 5봉 단위로 요약
        df_recent = df.tail(20) # 최근 흐름은 1봉 단위 (정밀 좌표용)
        df_old = df.iloc[:-20].tail(80).iloc[::5] # 과거 흐름은 요약 (추세 파악용)
        df_combined = pd.concat([df_old, df_recent])
        
        # GPT가 시간값을 잃어버리지 않게 포맷팅
        df_summary = df_combined[[time_col, 'high', 'low', close_col, 'rsi']].to_string(index=False)

        # 3) HTF 분석
        current_rsi = float(df['rsi'].iloc[-1])
        macro_info = "거시 상승 추세" if df[close_col].iloc[-1] > df[close_col].iloc[-50] else "횡보/하락"

        return analyze_with_gpt(data.symbol, data.current_price, {'current_rsi': current_rsi}, df_summary, data.question, macro_info)

    except Exception as e:
        logger.error(f"Global Error: {e}")
        return {"decision": "STAY", "reason": f"분석 오류: {str(e)[:50]}"}

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

# OpenAI 클라이언트 (Railway 환경변수 사용)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class MarketData(BaseModel):
    symbol: str
    current_price: float
    df_json: str
    question: str = "현재 시장 상황을 분석해줘"

def analyze_with_gpt(symbol, current_price, rsi_info, df_context, user_query, macro_info):
    # 시각화 좌표 추출 로직이 추가된 마스터 프롬프트
    prompt = f"""
    너는 HTF(거시 추세)를 읽고 '3-4-5파 하락 다이버전스'를 포착하는 'Short Specialist' Dober AI야.
    아래 가이드라인에 따라 사용자의 질문 "{user_query}"에 답변하고 시각화 좌표를 생성해.

    **[1. 핵심 분석 로직: 숏 타점]**
    - 3파 확인: RSI 70 이상 과매수와 함께 강한 상승을 보인 구간.
    - 4파 조정: 3파 이후 RSI가 식으며 이전 고점 RSI를 하향 돌파한 구간.
    - 5파 오버슈팅: 가격은 3파 고점을 넘었으나, RSI는 낮은 '하락 다이버전스' 발생 여부.
    
    **[2. 거시적 관점(HTF)]**: {macro_info}
    - 히든 상승 다이버전스 감지 시 숏 진입은 매우 보수적으로 판단.

    **[3. 데이터 정보]**
    - 종목: {symbol}, 현재가: {current_price}, RSI: {rsi_info['current_rsi']:.2f}
    - 최근 차트 샘플:
    {df_context}

    **[4. 출력 형식 (반드시 JSON)]**
    반드시 아래 구조를 유지해:
    {{
        "decision": "ENTER/STAY",
        "reason": "단계별 정밀 리포트 내용",
        "visual_data": {{
            "current_wave": "현재파동번호",
            "points": [
                {{"label": "1", "type": "high", "timestamp": "시간값", "price": "가격"}},
                {{"label": "2", "type": "low", "timestamp": "시간값", "price": "가격"}},
                {{"label": "3", "type": "high", "timestamp": "시간값", "price": "가격"}},
                {{"label": "4", "type": "low", "timestamp": "시간값", "price": "가격"}}
            ]
        }}
    }}
    *주의: timestamp는 제공된 샘플 데이터 내의 실제 시간값을 사용해.*
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "너는 엘리어트 파동 좌표 추출 전문가다. 무조건 JSON으로만 응답하라."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" }
        )
        res_json = json.loads(response.choices[0].message.content)
        
        # 'reason' 키값이 누락될 경우를 대비한 방어 로직
        if "reason" not in res_json:
            res_json["reason"] = "리포트 생성 중 규격 오류가 발생했습니다. 상세 데이터는 로그를 확인하세요."
        
        return res_json
    except Exception as e:
        logger.error(f"GPT Error: {e}")
        return {"decision": "STAY", "reason": "AI 분석 엔진 연산 지연 중.."}

@app.post("/api/v1/analyze-short")
async def run_strategy(data: MarketData):
    try:
        # 1) 데이터 로드 (최소 100개 캔들 권장)
        df = pd.read_json(io.StringIO(data.df_json))
        df.columns = [str(c).lower() for c in df.columns]
        close_col = next((c for c in df.columns if c in ['close', 'c', '4']), df.columns[-1])
        time_col = 'timestamp' if 'timestamp' in df.columns else df.columns[0]

        # 2) RSI 및 데이터 정제
        delta = df[close_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().replace(0, 0.001)
        df['rsi'] = 100 - (100 / (1 + (gain / loss)))
        df['rsi'] = df['rsi'].fillna(50)

        # 3) HTF 선행 분석
        current_rsi = float(df['rsi'].iloc[-1])
        macro_trend = "상승 추세" if df[close_col].iloc[-1] > df[close_col].iloc[-50] else "하락/횡보"
        
        recent_low_idx = df[close_col].tail(50).idxmin()
        hidden_div = "감지" if (df[close_col].iloc[-1] > df[close_col].loc[recent_low_idx] and 
                               current_rsi < df['rsi'].loc[recent_low_idx]) else "미감지"
        macro_info = f"거시적 {macro_trend} (히든 상승 다이버전스 {hidden_div})"

        # 4) 샘플링 (부하 방지 및 좌표 정확도 확보를 위해 15봉 샘플링)
        sampled_df = df.tail(100).iloc[::5] # 100봉을 5봉 간격으로 요약
        df_context = sampled_df[[time_col, 'high', close_col, 'rsi']].to_string(index=False)

        # 5) 최종 실행
        return analyze_with_gpt(data.symbol, data.current_price, {'current_rsi': current_rsi}, df_context, data.question, macro_info)

    except Exception as e:
        logger.error(f"Global Error: {e}")
        return {"decision": "STAY", "reason": f"분석 프로세스 오류: {str(e)[:50]}"}

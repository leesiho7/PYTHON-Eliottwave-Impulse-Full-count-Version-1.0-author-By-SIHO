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
    # 시호님의 'Short Specialist' 3-4-5파 로직을 주입한 프롬프트
    prompt = f"""
    너는 HTF(거시 추세)를 읽고 '3-4-5파 하락 다이버전스'를 잡아내는 'Short Specialist' Dober AI야.
    아래 가이드라인에 따라 사용자의 질문 "{user_query}"에 답변해줘.

    **[핵심 분석 로직: 숏 타점 포착]**
    1. **3파 확인**: 최근 구간 중 RSI가 70 이상(과매수)으로 치솟으며 강력한 상승을 보인 지점.
    2. **4파 조정**: 3파 이후 RSI가 식으면서 이전 RSI 고점을 하향 돌파한 지점.
    3. **5파 오버슈팅**: 가격은 3파 고점을 넘었으나, RSI는 3파 고점보다 낮은 '하락 다이버전스' 발생 여부.
    
    **[거시적 관점(HTF)]**: {macro_info}
    - 만약 '히든 상승 다이버전스'가 감지되었다면 숏 진입은 극도로 보수적으로 판단할 것.

    **[현재 데이터]** 종목: {symbol}, 가격: {current_price}, RSI: {rsi_info['current_rsi']:.2f}
    **[차트 흐름(샘플링)]**
    {df_context}

    반드시 아래 JSON 형식으로만 응답해:
    {{
        "decision": "ENTER/STAY",
        "reason": "1.현재 파동 위치, 2.로직 충족 여부(체크리스트), 3.거시 추세 반영 결론을 포함한 리포트"
    }}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "너는 엘리어트 파동과 RSI 다이버전스 전문가다. 무조건 JSON으로 응답하라."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" }
        )
        res_json = json.loads(response.choices[0].message.content)
        return res_json
    except Exception as e:
        logger.error(f"GPT Error: {e}")
        return {"decision": "STAY", "reason": "AI 분석 노드가 연산 중입니다. 잠시 후 다시 시도해주세요."}

@app.post("/api/v1/analyze-short")
async def run_strategy(data: MarketData):
    try:
        # 1) 데이터 로드 및 전처리 (자바에서 100개 이상의 캔들을 보낼 것을 권장)
        df = pd.read_json(io.StringIO(data.df_json))
        df.columns = [str(c).lower() for c in df.columns]
        close_col = next((c for c in df.columns if c in ['close', 'c', '4']), df.columns[-1])

        # 2) RSI 계산 (14 period)
        delta = df[close_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().replace(0, 0.001)
        df['rsi'] = 100 - (100 / (1 + (gain / loss)))
        df['rsi'] = df['rsi'].fillna(50)

        # 3) HTF(거시 추세) 및 히든 다이버전스 선행 분석
        current_rsi = float(df['rsi'].iloc[-1])
        # 50봉 전과 비교하여 거시 추세 판단
        macro_trend = "상승 추세" if df[close_col].iloc[-1] > df[close_col].iloc[-50] else "하락/횡보"
        
        # 히든 상승 다이버전스 (가격 저점은 높아지는데 RSI 저점은 낮아지는 현상)
        recent_low_idx = df[close_col].tail(50).idxmin()
        hidden_div = "감지(상승 힘 강함)" if (df[close_col].iloc[-1] > df[close_col].loc[recent_low_idx] and 
                                       current_rsi < df['rsi'].loc[recent_low_idx]) else "미감지"

        macro_info = f"거시적 {macro_trend} 속에서 히든 상승 다이버전스 징후가 {hidden_div}합니다."

        # 4) 데이터 샘플링 (부하 방지: 5봉 단위로 요약하여 GPT 전달)
        sampled_df = df.iloc[::5].tail(20) 
        df_context = sampled_df[['high', close_col, 'rsi']].to_string(index=False)

        # 5) 최종 분석 실행
        return analyze_with_gpt(data.symbol, data.current_price, {'current_rsi': current_rsi}, df_context, data.question, macro_info)

    except Exception as e:
        logger.error(f"Global Error: {e}")
        return {"decision": "STAY", "reason": f"시스템 분석 중 오류 발생: {str(e)[:50]}"}

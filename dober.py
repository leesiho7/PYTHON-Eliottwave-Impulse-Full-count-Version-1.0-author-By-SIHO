import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import json
import logging

# 1. 설정 및 모델 초기화
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Railway Variables에 등록한 OPENAI_API_KEY를 자동으로 가져옵니다.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 데이터 구조 정의
class MarketData(BaseModel):
    symbol: str
    current_price: float
    df_json: str  # 자바에서 전달한 캔들 데이터(JSON String)

# 2. 도메인 로직: RSI 계산 (데이터 안정성 보강)
def calculate_rsi(df, period=14):
    try:
        # 'close' 컬럼을 안전하게 찾기
        close_col = 'close' if 'close' in df.columns else df.columns[df.columns.str.lower() == 'close'][0]
        delta = df[close_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    except Exception as e:
        logger.error(f"RSI 계산 중 오류: {e}")
        return pd.Series([0] * len(df))

# 3. 에이전트 핵심 분석 함수
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
            model="gpt-4-turbo-preview", 
            messages=[
                {"role": "system", "content": "너는 금융 분석 전문가이며 JSON으로만 응답하는 봇이다."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" }
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
        # 자바에서 넘어오는 다양한 JSON 리스트 형식을 Pandas가 읽을 수 있도록 orient='records'나 기본 읽기 시도
        try:
            df = pd.read_json(data.df_json)
        except:
            # 리스트 형태의 JSON인 경우를 대비한 2차 시도
            import io
            df = pd.read_json(io.StringIO(data.df_json))

        # 2) 모든 컬럼명을 소문자로 통일 (가장 중요한 수정 사항)
        df.columns = [c.lower() for c in df.columns]

        # 3) 'close' 데이터 존재 여부 체크
        if 'close' not in df.columns:
            return {"decision": "STAY", "reason": f"Error: 'close' column missing. Found: {list(df.columns)}"}

        # 4) RSI 계산 및 분석 데이터 준비
        df['rsi'] = calculate_rsi(df)
        current_rsi = float(df['rsi'].iloc[-1])
        max_rsi_3p = float(df['rsi'].tail(50).max())
        
        # 5) 필터링: RSI가 너무 낮으면 AI 호출 없이 종료
        if current_rsi < 60:
            return {"decision": "STAY", "reason": f"RSI too low ({current_rsi:.2f})"}

        # 6) 최근 20개 캔들 요약
        # GPT에게 보낼 때 불필요한 인덱스 제외하고 주요 데이터만 추출
        recent_context = df.tail(20)[['high', 'close', 'rsi']].to_string(index=False)
        rsi_info = {'current_rsi': current_rsi, 'max_rsi_3p': max_rsi_3p}

        # 7) GPT 에이전트 분석 호출
        final_analysis = analyze_with_gpt(data.symbol, data.current_price, rsi_info, recent_context)
        
        return final_analysis

    except Exception as e:
        logger.error(f"Global Error: {str(e)}")
        return {"decision": "STAY", "reason": f"Server Error: {str(e)}"}
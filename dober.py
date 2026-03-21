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

def calculate_rsi(df, period=14):
    try:
        # close 컬럼 찾기 (대소문자 무시 및 숫자형 컬럼 방어)
        target_col = None
        for col in df.columns:
            if str(col).lower() == 'close':
                target_col = col
                break
        
        if target_col is None:
            return pd.Series([0] * len(df))

        delta = df[target_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    except Exception as e:
        return pd.Series([0] * len(df))

def analyze_with_gpt(symbol, current_price, rsi_info, df_context):
    prompt = f"""
    너는 엘리어트 파동과 RSI 다이버전스 분석 전문가야.
    [데이터] 종목:{symbol}, 현재가:{current_price}, RSI:{rsi_info['current_rsi']:.2f}
    [흐름] {df_context}
    반드시 JSON으로만 답해: {{ "decision": "ENTER/STAY", "reason": "한줄", "entry_price": {current_price}, "stop_loss": 0, "take_profit": 0 }}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except:
        return {"decision": "STAY", "reason": "AI Error"}

@app.post("/api/v1/analyze-short")
async def run_strategy(data: MarketData):
    try:
        # 1) JSON 읽기 (예외처리 강화)
        try:
            df = pd.read_json(data.df_json)
        except:
            import io
            df = pd.read_json(io.StringIO(data.df_json))

        # 2) 컬럼명 처리 (숫자가 들어와도 문자열로 바꿔서 lower() 실행)
        # 여기서 'int' object has no attribute 'lower' 에러를 잡습니다.
        df.columns = [str(c).lower() for c in df.columns]

        # 3) 필수 데이터 확인
        if 'close' not in df.columns:
            return {"decision": "STAY", "reason": f"데이터 필드명 불일치 (받은 항목: {list(df.columns)})"}

        # 4) 계산 및 분석
        df['rsi'] = calculate_rsi(df)
        current_rsi = float(df['rsi'].iloc[-1])
        
        if current_rsi < 60:
            return {"decision": "STAY", "reason": f"RSI 강도 부족 ({current_rsi:.2f})"}

        recent_context = df.tail(15)[['high', 'close', 'rsi']].to_string(index=False)
        rsi_info = {'current_rsi': current_rsi, 'max_rsi_3p': float(df['rsi'].max())}

        return analyze_with_gpt(data.symbol, data.current_price, rsi_info, recent_context)

    except Exception as e:
        return {"decision": "STAY", "reason": f"System Check: {str(e)}"}
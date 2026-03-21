# 1. 가벼운 파이썬 이미지 사용 (메모리 다이어트)
FROM python:3.9-slim

# 2. 필수 패키지 설치
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. 소스 코드 복사
COPY . .

# 4. 포트 설정 및 uvicorn 실행 (Worker 수를 조절하여 안정성 확보)
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
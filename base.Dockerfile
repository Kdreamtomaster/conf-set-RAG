# Python 3.11 기반 이미지 사용
FROM python:3.11

# 작업 디렉토리 설정
WORKDIR /app

# 종속성 파일 복사 및 설치 (여기까지 캐싱 가능)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

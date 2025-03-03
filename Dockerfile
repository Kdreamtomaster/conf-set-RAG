FROM myapp-base

# 작업 디렉토리 설정
WORKDIR /app


# 앱 소스 코드 복사
COPY . .

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1

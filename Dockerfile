# =====================================================================================
# STAGE 1: Build Stage - 모든 라이브러리를 설치하고 빌드하는 단계
# =====================================================================================
FROM python:3.9-slim as builder

# 시스템 패키지 설치 (libffi-dev, build-essential 등은 cffi, cryptography 등 일부 패키지 빌드에 필요)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libffi-dev \
    build-essential \
    mecab \
    default-jdk \
    g++ \
    curl \
    git \
    fonts-nanum \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# pip 업그레이드
RUN pip install --upgrade pip

# =====================================================================================
# Python 라이브러리 설치 (충돌을 피하기 위한 최적의 순서)
# =====================================================================================

# 1. PyTorch 1.10.1 설치 (사용자 요구사항)
RUN pip install torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1+cpu -f https://download.pytorch.org/whl/cpu

# 2. PyTorch 1.10.1과 호환되는 transformers 버전을 직접 설치 (핵심 수정 사항)
RUN pip install transformers==4.28.1

# 3. KoBERT 토크나이저에 필요한 sentencepiece 설치
RUN pip install sentencepiece

# 4. requirements.txt 복사 및 설치 (transformers 관련 라인은 삭제된 상태여야 함)
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Whisper 설치
RUN pip install --no-cache-dir git+https://github.com/openai/whisper.git@v20230124


# =====================================================================================
# STAGE 2: Final Stage - 실제 실행에 필요한 것들만 담는 최종 단계
# =====================================================================================
FROM python:3.9-slim

# 실행에 필요한 최소한의 시스템 패키지만 설치
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    mecab \
    default-jdk \
    fonts-nanum \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# 환경 변수 설정
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Build Stage에서 설치한 Python 라이브러리들만 복사
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

# 작업 디렉토리 설정
WORKDIR /app

# 소스 코드 복사
COPY . .

# 필요한 폴더 미리 생성
RUN mkdir -p uploads \
    static/csv \
    deepvoice_detection/model \
    KoBERTModel/model

# Flask 포트 오픈
EXPOSE 5000

# Gunicorn WSGI 서버로 Flask 앱 실행 (워커 2개)
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "--timeout", "360", "server:app"]
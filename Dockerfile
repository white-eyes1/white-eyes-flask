# syntax=docker/dockerfile:1
# 분석 서버 이미지. 의존성은 uv.lock 그대로 설치한다(torch는 Linux에서 CPU 전용 저장소)
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.22 /uv /usr/local/bin/uv
# ffmpeg: 라이브 녹화(FNC-S-008), libgl1·libglib2.0-0: opencv-python, curl: compose healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 curl \
 && rm -rf /var/lib/apt/lists/*
ENV TZ=Asia/Seoul \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev
ENV PATH=/app/.venv/bin:$PATH
EXPOSE 5000
CMD ["white-eyes-flask"]

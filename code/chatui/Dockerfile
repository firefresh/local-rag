FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN apt update && apt install -y curl
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir opencv-python-headless==4.9.0.80

COPY ./src /app/src
COPY ./config /app/config

ENV SENTENCE_TRANSFORMERS_HOME=/models/cache
RUN mkdir -p /models/cache
VOLUME [ "/models/cache" ]

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 CMD [ "curl", "-f", "http://localhost:8080/" ]

ENV PROXY_PREFIX=

ENTRYPOINT [ "python3", "-m", "src", "--config", "/app/config/chat.yaml" ]
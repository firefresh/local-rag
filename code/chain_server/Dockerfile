FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN apt update && apt install -y curl
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir opencv-python-headless==4.9.0.80

COPY ./src /app/src
COPY ./config /app/config
COPY entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

# Set environment variables
ENV PROXY_PREFIX=
# Hugging Face data location
ENV HUGGINGFACE_HUB_CACHE=/models/cache
ENV SENTENCE_TRANSFORMERS_HOME=/models/cache
ENV APP_CONFIG_FILE=/app/config/chain.yaml

RUN mkdir -p /models/cache
VOLUME [ "/models/cache" ]

EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 CMD [ "curl", "-f", "http://localhost:10000/health" ]

ENTRYPOINT [ "/app/entrypoint.sh" ]
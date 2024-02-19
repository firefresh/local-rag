#!/bin/bash

# Start API
python3 src/common/init_tests.py \
    && python3 -m uvicorn src.server:app --port=10000 --host='0.0.0.0' &
    pid1=$!

if ! wait $pid1; then
    echo "API failed"
    exit 1
fi

echo "RAG system ready"
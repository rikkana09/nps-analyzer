#!/bin/bash
set -e

echo "=========================================="
echo "  NPS Analyzer — полный пайплайн"
echo "=========================================="

echo ""
echo "[1/4] Парсинг xlsx..."
stdbuf -oL -eL python -u src/preprocess.py

echo ""
echo "[2/4] LLM-анализ (с трейсами в Langfuse)..."
stdbuf -oL -eL python -u src/run_all.py

echo ""
echo "[3/4] Построение FAISS-индекса..."
stdbuf -oL -eL python -u src/build_rag_index.py

echo ""
echo "[4/4] Запуск дашборда на http://localhost:8501"
echo "=========================================="
exec streamlit run app/dashboard.py \
    --server.address=0.0.0.0 \
    --server.port=8501
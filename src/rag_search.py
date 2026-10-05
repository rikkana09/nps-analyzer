"""
Поиск похожих отзывов через FAISS.
"""
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


INDEX_PATH = Path("data/processed/faiss_index.bin")
META_PATH = Path("data/processed/rag_metadata.parquet")
MODEL_NAME = "intfloat/multilingual-e5-small"


@lru_cache(maxsize=1)
def _load_resources():
    """Загрузка модели, индекса и метаданных (один раз)."""
    model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(str(INDEX_PATH))
    meta = pd.read_parquet(META_PATH)
    return model, index, meta


def find_similar(query: str, k: int = 5) -> pd.DataFrame:
    """
    Находит k наиболее похожих отзывов на запрос.
    Возвращает DataFrame с колонками:
    row_id, text, sentiment, topic, summary, score.
    """
    model, index, meta = _load_resources()

    # E5-модели требуют префикс "query: " для запросов
    query_text = "query: " + query
    emb = model.encode(
        [query_text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    # Поиск: возвращает расстояния и индексы
    scores, indices = index.search(emb, k)

    # Достаём метаданные по индексам
    results = meta.iloc[indices[0]].copy()
    results["score"] = scores[0]
    return results.reset_index(drop=True)


if __name__ == "__main__":
    queries = [
        "Дорогой тариф, слишком высокая цена",
        "Не работает интернет дома",
        "Спасибо большое, всё отлично",
    ]
    for q in queries:
        print("=" * 70)
        print(f"ЗАПРОС: {q}")
        print("=" * 70)
        df = find_similar(q, k=3)
        for _, row in df.iterrows():
            print(f"  [score={row['score']:.3f}] {row['sentiment']} | {row['topic']}")
            print(f"  Текст: {row['text'][:100]}")
            print(f"  Резюме: {row['summary']}")
            print()

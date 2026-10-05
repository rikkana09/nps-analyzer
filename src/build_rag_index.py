"""
Создание FAISS-индекса из эмбеддингов всех отзывов.
Модель: intfloat/multilingual-e5-small (384 размерности).
"""
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


INPUT_PATH = Path("data/processed/llm_results.parquet")
PARSED_PATH = Path("data/processed/nps_parsed.parquet")
INDEX_PATH = Path("data/processed/faiss_index.bin")
META_PATH = Path("data/processed/rag_metadata.parquet")
MODEL_NAME = "intfloat/multilingual-e5-small"


def main():
    print("Загрузка данных...")
    llm_df = pd.read_parquet(INPUT_PATH)
    parsed_df = pd.read_parquet(PARSED_PATH)

    df = llm_df.merge(
        parsed_df[["row_id", "text"]],
        on="row_id",
        how="left",
    )

    mask = df["text"].notna() & (df["text"].str.len() > 5)
    df = df[mask].copy()
    print(f"Текстов для индексации: {len(df)}")

    print(f"Загрузка модели {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    texts = ["passage: " + t for t in df["text"].tolist()]

    print("Создание эмбеддингов...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings).astype("float32")
    print(f"Размерность: {embeddings.shape}")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    print(f"Индекс создан: {index.ntotal} векторов")

    faiss.write_index(index, str(INDEX_PATH))
    print(f"Индекс: {INDEX_PATH}")

    meta_cols = ["row_id", "text", "sentiment", "topic",
                 "summary", "is_meaningful", "nps_score"]
    meta_cols = [c for c in meta_cols if c in df.columns]
    df[meta_cols].to_parquet(META_PATH, index=False)
    print(f"Метаданные: {META_PATH}")


if __name__ == "__main__":
    main()
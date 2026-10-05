"""
Создание и запуск дашборда
"""

import ast
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


DATA_PATH = Path("data/processed/llm_results.parquet")

st.set_page_config(page_title="NPS Analyzer", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_PATH)

    def to_list(x):
        """Приводит к list: строки, numpy-массивы, None."""
        if x is None:
            return []
        if isinstance(x, str):
            try:
                v = ast.literal_eval(x)
                return list(v) if hasattr(v, "__iter__") else []
            except Exception:
                return []
        if hasattr(x, "__iter__"):
            return list(x)
        return []

    def to_dict(x):
        """Приводит к dict."""
        if x is None:
            return {}
        if isinstance(x, str):
            try:
                v = ast.literal_eval(x)
                return v if isinstance(v, dict) else {}
            except Exception:
                return {}
        if isinstance(x, dict):
            return x
        return {}

    df["emotions"] = df["emotions"].apply(to_list)
    df["entities"] = df["entities"].apply(to_dict)
    return df

def main():
    st.title("📊 Анализ NPS-опросов")
    st.markdown("Автоматический разбор отзывов клиентов")

    if not DATA_PATH.exists():
        st.error(f"Файл {DATA_PATH} не найден. Сначала запусти `python src/run_all.py`.")
        return

    df = load_data()
    df_valid = df[df["sentiment"].notna()].copy()

    # --- Верхняя панель метрик ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Всего отзывов", len(df_valid))
    with col2:
        meaningful = df_valid["is_meaningful"].sum()
        st.metric("Содержательных", int(meaningful))
    with col3:
        neg_share = (df_valid["sentiment"] == "negative").mean() * 100
        st.metric("Доля негатива", f"{neg_share:.1f}%")
    with col4:
        pos_share = (df_valid["sentiment"] == "positive").mean() * 100
        st.metric("Доля позитива", f"{pos_share:.1f}%")

    st.divider()

    # --- Тональность ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Тональность")
        sent_counts = df_valid["sentiment"].value_counts()
        fig = px.pie(
            values=sent_counts.values,
            names=sent_counts.index,
            color=sent_counts.index,
            color_discrete_map={
                "positive": "#2ecc71",
                "neutral": "#95a5a6",
                "negative": "#e74c3c",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Темы обращений")
        topic_counts = df_valid["topic"].value_counts().head(10)
        fig = px.bar(
            x=topic_counts.values,
            y=topic_counts.index,
            orientation="h",
            labels={"x": "Количество", "y": ""},
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- Эмоции ---
    st.subheader("Эмоции")
    all_emotions = [e for lst in df_valid["emotions"] for e in lst]
    emotion_counts = Counter(all_emotions).most_common(10)
    if emotion_counts:
        labels, values = zip(*emotion_counts)
        fig = px.bar(x=list(labels), y=list(values), labels={"x": "", "y": "Количество"})
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- Топ проблем ---
    st.subheader("Топ проблем")
    all_problems = []
    for entities in df_valid["entities"]:
        if isinstance(entities, dict):
            all_problems.extend(entities.get("проблема", []))
    problem_counts = Counter(all_problems).most_common(15)
    if problem_counts:
        labels, values = zip(*problem_counts)
        fig = px.bar(x=list(values), y=list(labels), orientation="h",
                     labels={"x": "Количество", "y": ""})
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- Таблица отзывов ---
    st.subheader("Отзывы")
    sentiment_filter = st.multiselect(
        "Фильтр по тональности",
        options=["positive", "neutral", "negative"],
        default=["negative"],
    )
    filtered = df_valid[df_valid["sentiment"].isin(sentiment_filter)]
    st.dataframe(
        filtered[["row_id", "sentiment", "topic", "summary"]].head(100),
        use_container_width=True,
    )

    st.divider()

    # --- RAG: поиск похожих отзывов ---
    st.subheader("🔍 Поиск похожих отзывов")
    st.markdown(
        "Введите текст или запрос — система найдёт самые похожие отзывы "
        "из исторических данных (FAISS + multilingual-e5-small)."
    )

    query = st.text_input(
        "Запрос",
        placeholder="например: дорогой тариф, не работает интернет",
    )
    k = st.slider("Сколько похожих найти", min_value=3, max_value=10, value=5)

    if query and query.strip():
        with st.spinner("Ищу похожие..."):
            try:
                from rag_search import find_similar
                similar_df = find_similar(query, k=k)
                st.success(f"Найдено {len(similar_df)} похожих отзывов")
                for _, row in similar_df.iterrows():
                    score = row["score"]
                    with st.expander(
                        f"[{score:.3f}] {row['sentiment']} | {row['topic']}"
                    ):
                        st.markdown(f"**Текст:** {row['text']}")
                        st.markdown(f"**Резюме:** {row['summary']}")
                        st.caption(f"row_id: {row['row_id']}")
            except Exception as e:
                st.error(f"Ошибка поиска: {e}")

if __name__ == "__main__":
    main()

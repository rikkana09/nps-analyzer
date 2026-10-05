"""
Парсер xlsx с NPS-опросами
"""
import re
from pathlib import Path

import pandas as pd


INPUT_XLSX = "data/raw/result_nps.xlsx"
OUTPUT_PARQUET = "data/processed/nps_parsed.parquet"
MARK_COLS = ["mark_2", "mark_3", "mark_4", "mark_5", "mark_6", "mark_7"]


def clean_text(s: str) -> str:
    """Убирает лишние пробелы, двойные точки, мусор по краям."""
    s = str(s).strip()
    # Двойные/тройные точки в конце
    s = re.sub(r"\.{2,}$", ".", s)
    # Множественные пробелы
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def extract_text(row: pd.Series) -> str | None:
    """Собирает непустые значения из mark_2..mark_7 в один текст."""
    parts = []
    for col in MARK_COLS:
        val = row.get(col)
        if pd.isna(val):
            continue
        s = clean_text(val)
        if s and s not in parts:  # не дублируем одинаковые фрагменты
            parts.append(s)
    if not parts:
        return None
    return " ".join(parts)


def main():
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    print(f"Чтение {INPUT_XLSX}...")
    df = pd.read_excel(INPUT_XLSX, sheet_name="Лист2")
    print(f"Строк в xlsx: {len(df)}")

    records = []
    for idx, row in df.iterrows():
        text = extract_text(row)
        records.append({
            "row_id": idx,
            "create_date": row.get("create_date"),
            "text": text,
        })

    result = pd.DataFrame(records)

    # Статистика
    n_total = len(result)
    n_text = result["text"].notna().sum()
    print(f"\nВсего строк: {n_total}")
    print(f"С текстом:   {n_text}")
    print(f"Пустых:      {n_total - n_text}")

    # Примеры
    print("\nПримеры текстов:")
    for t in result["text"].dropna().sample(min(5, n_text)):
        print(f"  • {t[:100]}")

    # Сохранение
    result.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\nСохранено: {OUTPUT_PARQUET}")


if __name__ == "__main__":
    main()

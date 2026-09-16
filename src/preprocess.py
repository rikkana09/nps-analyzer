import re
import pandas as pd
from pathlib import Path


def is_number_like(s: str) -> bool:
    """Проверяет, состоит ли строка только из цифр, точек и запятых."""
    s = s.strip().replace(" ", "")
    return bool(re.fullmatch(r"[\d,.]+", s))


def parse_number(s: str):
    """
    Всегда возвращает список целых чисел.
    '5, 0'    → [5, 0]
    '2 , 9'   → [2, 9]
    '2,3,5.'  → [2, 3, 5]
    '025789'  → [0, 2, 5, 7, 8, 9]
    '9'       → [9]
    '1 5'     → [1, 5]
    '2 тариф' → None
    """
    s = s.strip().replace(" ", "")
    if not re.fullmatch(r"[\d,.]+", s):
        return None
    s = s.rstrip(".")
    if re.search(r"[.,]", s):
        parts = [p for p in re.split(r"[,.]+", s) if p]
        try:
            return [int(p) for p in parts]
        except ValueError:
            return None
    if len(s) > 1:
        return [int(ch) for ch in s]
    return [int(s)]


def extract_nps(numbers: list) -> tuple:
    """
    Из списка чисел извлекает NPS-балл и категорию.
    Гипотеза: NPS — первое число в списке.
    """
    if not numbers:
        return None, None
    score = numbers[0]
    if not (0 <= score <= 10):
        return None, None
    if score >= 9:
        category = "promoter"
    elif score >= 7:
        category = "neutral"
    else:
        category = "detractor"
    return score, category


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Лист2")
    df = df.dropna(how="all")
    return df


def extract_answers(row: pd.Series) -> list:
    answers = []
    for col in ["mark_2", "mark_3", "mark_4", "mark_5", "mark_6", "mark_7"]:
        val = row.get(col)
        if pd.isna(val):
            continue
        s = str(val).strip()
        if not s:
            continue
        if is_number_like(s):
            parsed = parse_number(s)
            if parsed:
                answers.append({
                    "source": col,
                    "raw": s,
                    "type": "numbers",
                    "value": parsed,
                })
            else:
                answers.append({
                    "source": col,
                    "raw": s,
                    "type": "text",
                    "value": None,
                })
        else:
            answers.append({
                "source": col,
                "raw": s,
                "type": "text",
                "value": None,
            })
    return answers


def build_dataset(path: str) -> pd.DataFrame:
    df = load_raw(path)
    records = []
    for idx, row in df.iterrows():
        answers = extract_answers(row)
        texts = [a["raw"] for a in answers if a["type"] == "text"]
        numbers = []
        for a in answers:
            if a["type"] == "numbers":
                numbers.extend(a["value"])
        nps_score, nps_category = extract_nps(numbers)
        records.append({
            "row_id": idx,
            "create_date": row.get("create_date"),
            "text": " ".join(texts) if texts else None,
            "numbers": numbers,
            "nps_score": nps_score,
            "nps_category": nps_category,
        })
    return pd.DataFrame(records)


if __name__ == "__main__":
    raw_path = "data/raw/result_nps_all.xlsx"
    out_path = "data/processed/nps_parsed.parquet"
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    df = build_dataset(raw_path)
    df.to_parquet(out_path, index=False)

    print(f"Готово. Строк: {len(df)}")
    print(f"С текстом: {df['text'].notna().sum()}")
    print(f"С NPS: {df['nps_score'].notna().sum()}")
    print(f"\nПервые 5 строк:")
    print(df.head())

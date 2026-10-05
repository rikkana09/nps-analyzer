"""
Батчевая обработка всех текстов через LLM.
Поддерживает:
- повторные попытки при ошибках
- чекпоинты (сохранение каждые N текстов)
- resume (пропускает уже успешно обработанные)
- трейсинг всех LLM-запросов в Langfuse
"""
import time
from pathlib import Path
import os
import ollama
_client = ollama.Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
import pandas as pd
from dotenv import load_dotenv
from langfuse import observe, get_client
from tqdm import tqdm

from llm_extract import (
    load_prompt_template,
    parse_json_response,
    normalize_result,
    MODEL,
)


# --- Конфигурация ---
load_dotenv()
langfuse = get_client()

INPUT_PATH = Path("data/processed/nps_parsed.parquet")
OUTPUT_PATH = Path("data/processed/llm_results.parquet")
CHECKPOINT_EVERY = 50
MAX_RETRIES = 3
RETRY_DELAY = 2


# --- LLM с трейсингом ---
@observe(name="nps-extract")
def extract_via_sdk(text: str, template: str, row_id: int) -> dict:
    """Один LLM-вызов. Автоматически трейсится в Langfuse."""
    prompt = template.replace("{text}", text)
    response = _client.generate(
        model=MODEL,
        prompt=prompt,
        options={"temperature": 0.1, "num_predict": 2000},
        keep_alive="10m",
        think=False,
    )
    raw = response["response"]
    result = normalize_result(parse_json_response(raw))
    result["row_id"] = row_id

    # Пишем input/output в текущий трейс Langfuse
    try:
        langfuse.update_current_span(
            input={"text": text, "row_id": row_id},
            output=result,
        )
    except AttributeError:
        try:
            langfuse.set_current_trace_io(
                input={"text": text, "row_id": row_id},
                output=result,
            )
        except AttributeError:
            pass

    return result

def extract_with_retry(text: str, template: str, row_id: int) -> dict:
    """3 попытки с задержкой."""
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            return extract_via_sdk(text, template, row_id)
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    raise last_err


def empty_result(row_id: int, error: str) -> dict:
    return {
        "row_id": row_id,
        "is_meaningful": None,
        "sentiment": None,
        "sentiment_confidence": None,
        "emotions": [],
        "entities": {},
        "topic": None,
        "summary": None,
        "error": str(error)[:500],
    }


def main():
    df = pd.read_parquet(INPUT_PATH)
    mask = df["text"].notna() & (df["text"].str.len() > 3)
    df_to_process = df[mask].copy()
    print(f"Всего текстов для обработки: {len(df_to_process)}")

    # Resume: пропускаем уже успешно обработанные
    if OUTPUT_PATH.exists():
        old_df = pd.read_parquet(OUTPUT_PATH)
        success_df = old_df[old_df["sentiment"].notna()].copy()
        done_ids = set(success_df["row_id"].tolist())
        results = success_df.to_dict("records")
        print(f"Уже успешно обработано: {len(done_ids)}")
        print(f"Осталось:               {len(df_to_process) - len(done_ids)}")
    else:
        done_ids = set()
        results = []

    template = load_prompt_template()
    errors_count = 0
    since_checkpoint = 0
    t0 = time.time()

    for _, row in tqdm(df_to_process.iterrows(), total=len(df_to_process)):
        row_id = int(row["row_id"])
        if row_id in done_ids:
            continue

        try:
            result = extract_with_retry(row["text"], template, row_id)
            results.append(result)
        except Exception as e:
            errors_count += 1
            results.append(empty_result(row_id, str(e)))

        since_checkpoint += 1
        if since_checkpoint >= CHECKPOINT_EVERY:
            pd.DataFrame(results).to_parquet(OUTPUT_PATH, index=False)
            since_checkpoint = 0

    pd.DataFrame(results).to_parquet(OUTPUT_PATH, index=False)
    elapsed = time.time() - t0
    print(f"\nГотово за {elapsed / 60:.1f} мин")
    print(f"Всего в файле: {len(results)}")
    print(f"Успешных:      {sum(1 for r in results if r.get('sentiment'))}")
    print(f"Новых ошибок:  {errors_count}")


if __name__ == "__main__":
    main()

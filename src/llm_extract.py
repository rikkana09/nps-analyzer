"""
взаимодействие с LLM
"""
import json
import re
import subprocess
from pathlib import Path


PROMPT_PATH = Path("prompts/extract.txt")
MODEL = "qwen3:8b"

#допустимые значения тональности, эмоций, тем
ALLOWED_SENTIMENTS = {"positive", "neutral", "negative"}
ALLOWED_EMOTIONS = {
    "радость", "благодарность", "нейтрально",
    "раздражение", "гнев", "разочарование",
    "тревога", "усталость",
}
ALLOWED_TOPICS = {
    "связь и покрытие", "тарифы и оплата",
    "обслуживание и операторы", "приложение и личный кабинет",
    "подключение и установка", "прочее",
}

# ANSI escape-последовательности (артефакты CLI Ollama)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# Маппинг русских вариантов на английские
SENTIMENT_MAP = {
    "позитивный": "positive", "позитивно": "positive",
    "нейтральный": "neutral", "нейтрально": "neutral",
    "негативный": "negative", "негативно": "negative",
}


def load_prompt_template() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def call_ollama(prompt: str, model: str = MODEL) -> str:
    """Синхронный вызов ollama через CLI (используется только для тестов)."""
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Ollama error: {result.stderr}")
    return result.stdout.strip()


def _repair_truncated_json(s: str) -> str:
    """Пытается закрыть обрезанный JSON: добавляет кавычки и скобки."""
    # Убираем висящую запятую в конце
    s = re.sub(r",\s*$", "", s.rstrip())
    # Баланс кавычек
    if s.count('"') % 2 == 1:
        s += '"'
    # Баланс скобок
    open_braces = s.count("{") - s.count("}")
    open_brackets = s.count("[") - s.count("]")
    s += "]" * max(open_brackets, 0)
    s += "}" * max(open_braces, 0)
    return s


def parse_json_response(raw: str) -> dict:
    """Достаёт первый валидный JSON-объект из ответа модели.
    Устойчив к markdown, ANSI-мусору и обрезанным ответам."""
    raw = ANSI_RE.sub("", raw)
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.replace("\r", " ").replace("\t", " ")

    decoder = json.JSONDecoder(strict=False)

    # 1. Ищем валидный JSON среди всех вхождений "{"
    for i, ch in enumerate(raw):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(raw[i:].replace("\n", " "))
                if isinstance(obj, dict) and "sentiment" in obj:
                    return obj
            except json.JSONDecodeError:
                continue

    # 2. Если не нашли — пробуем починить обрезанный JSON
    start = raw.find("{")
    if start == -1:
        raise ValueError(f"JSON not found in response: {raw[:300]}")
    candidate = raw[start:].replace("\n", " ")
    candidate = _repair_truncated_json(candidate)
    try:
        obj = json.loads(candidate, strict=False)
        if isinstance(obj, dict) and "sentiment" in obj:
            return obj
    except json.JSONDecodeError:
        pass

    raise ValueError(f"JSON not found in response: {raw[:300]}")


def normalize_result(result: dict) -> dict:
    """Приводит ответ модели к строгой схеме."""
    # sentiment
    sentiment = str(result.get("sentiment", "neutral")).lower().strip()
    sentiment = SENTIMENT_MAP.get(sentiment, sentiment)
    if sentiment not in ALLOWED_SENTIMENTS:
        sentiment = "neutral"
    result["sentiment"] = sentiment

    # emotions
    emotions = result.get("emotions", [])
    if isinstance(emotions, str):
        emotions = [emotions]
    result["emotions"] = [e for e in emotions if e in ALLOWED_EMOTIONS][:2]

    # topic
    topic = result.get("topic", "прочее")
    if topic not in ALLOWED_TOPICS:
        topic = "прочее"
    result["topic"] = topic

    # is_meaningful
    result["is_meaningful"] = bool(result.get("is_meaningful", True))

    # entities — все поля должны быть
    entities = result.get("entities", {})
    if not isinstance(entities, dict):
        entities = {}
    for key in ["продукт", "проблема", "сотрудник", "локация"]:
        if key not in entities:
            entities[key] = []
    result["entities"] = entities

    # confidence
    try:
        result["sentiment_confidence"] = float(result.get("sentiment_confidence", 0.5))
    except (ValueError, TypeError):
        result["sentiment_confidence"] = 0.5

    # summary
    result["summary"] = str(result.get("summary", "")).strip()

    return result


def extract(text: str) -> dict:
    """Полный пайплайн: промпт → ollama → парсинг → нормализация."""
    template = load_prompt_template()
    prompt = template.replace("{text}", text)
    raw = call_ollama(prompt)
    result = parse_json_response(raw)
    return normalize_result(result)


if __name__ == "__main__":
    test_texts = [
        "Оператору вообще уже 3 года не можем дозвониться",
        "Спасибо",
        "Изменение должно быть комплексным. От изменения зоны покрытия до работы с тарифами.",
        "Очень дорого, снизьте абонентскую плату",
        "Всё хорошо, спасибо за работу!",
    ]
    for t in test_texts:
        print("=" * 60)
        print(f"TEXT: {t}")
        try:
            result = extract(t)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"ERROR: {e}")
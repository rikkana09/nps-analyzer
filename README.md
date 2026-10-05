# NPS Analyzer. Автоматизация разбора NPS-опросов клиентов

**Автор:** Нина Шульга
**Дата:** сентябрь-октябрь 2026
**Стек:** Python, Ollama (qwen3:8b), Hugging Face, Streamlit, FAISS, Langfuse

---

## Задача

NPS-опросы клиентов компании приходят в виде xlsx-файлов, где текстовые комментарии и числовые ответы перемешаны. Ручной разбор 1000+ анкет занимает дни и не даёт системной картины. Требуется автоматизировать анализ.

## Цель проекта

Разработать автоматизированный пайплайн анализа текстовых NPS-опросов с помощью локальной LLM, формирующий дашборд с оценкой каждой анкеты и агрегированной аналитикой, который позволяет искать похожие отзывы (RAG). Обеспечить **LLM Observability через Langfuse** — трейсинг всех запросов, замеры latency и токенов для контроля качества работы модели.

## Что делает проект

Принимает xlsx-файл с NPS-опросом (1347 текстовых анкет), анализирует каждый комментарий через локальную LLM и формирует дашборд с:

- **Тональностью** — positive / neutral / negative
- **Эмоциями** — разочарование, раздражение, благодарность, радость и др.
- **Сущностями (NER)** — проблемы, продукты, сотрудники, локации
- **Темой** — одна из 6 фиксированных категорий
- **Резюме** — краткое описание отзыва

Дополнительно:

- **RAG-поиск** похожих отзывов через FAISS
- **LLM Observability** через Langfuse (трейсы, latency, токены)
- **Docker** для воспроизводимого развёртывания
- **Интерактивный дашборд** на Streamlit

## Что на входе / выходе

- **Вход:** xlsx-файл с NPS-опросом (~1347 строк). Ответы размазаны по 6 столбцам.
- **Выход:** дашборд с оценкой каждой анкеты + агрегированные метрики.


## Архитектура

result_nps.xlsx
↓ preprocess.py
nps_parsed.parquet
↓ run_all.py (Ollama: Qwen 3, → Langfuse)
llm_results.parquet
↓ build_rag_index.py (multilingual-e5-small)
faiss_index.bin + rag_metadata.parquet
↓ dashboard.py (Streamlit)
Веб-интерфейс на :8501 порту

### Компоненты

| Компонент | Технология |
|---|---|
| LLM | Qwen 3 (8B) через Ollama — локальный GPU-инференс |
| Embeddings | intfloat/multilingual-e5-small |
| Векторный поиск | FAISS |
| Дашборд | Streamlit + Plotly |
| Observability | Langfuse Cloud |
| Данные | pandas, openpyxl, PyArrow (Parquet) |
| Валидация | scikit-learn |
| Развёртывание | Docker + docker-compose |

## Структура проекта

```
nps-analyzer/
├── app/
│ └── dashboard.py # Streamlit-дашборд
├── src/
│ ├── preprocess.py # парсер xlsx → parquet
│ ├── llm_extract.py # промпт + парсинг JSON
│ ├── run_all.py # батчевая LLM-обработка + Langfuse
│ ├── build_rag_index.py # создание FAISS-индекса
│ └── rag_search.py # поиск похожих отзывов
├── prompts/
│ └── extract.txt # шаблон промпта для LLM
├── data/
│ ├── raw/ # исходный xlsx
│ └── processed/ # parquet + FAISS-индекс
├── reports/
│ ├── report.md # отчёт проекта
│ └── screenshots/ # скриншоты дашборда и langfuse
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

## Запуск

### Предварительные требования

- Python 3.10+
- [Docker](https://docs.docker.com/) + Docker Compose
- [Ollama](https://ollama.com) с моделью `qwen3:8b`, запущенная на хосте
- GPU NVIDIA (опционально, для скорости)

### 1. Установка Ollama и модели

```bash
# Установить Ollama (см. ollama.com)
ollama pull qwen3:8b

# Убедиться, что Ollama слушает на всех интерфейсах
sudo systemctl edit ollama
# В открывшемся файле добавить:
# [Service]
# Environment="OLLAMA_HOST=0.0.0.0:11434"
sudo systemctl daemon-reload
sudo systemctl restart ollama

# Проверить
curl http://0.0.0.0:11434/api/tags
```

### 2. Клонирование и настройка

```bash
git clone https://github.com/rikkana09/nps-analyzer.git
cd nps-analyzer

# Копируем пример env
cp .env.example .env
# Редактируем .env — вставляем ключи Langfuse
```

**Что в `.env`:**

```env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 3. Запуск всего проекта одной командой

```bash
docker-compose up --build
```

**Что произойдёт в одном контейнере:**

```
[1/4] Парсинг xlsx...              → nps_parsed.parquet
[2/4] LLM-анализ (Qwen 3 + Langfuse) → llm_results.parquet (~50 мин)
[3/4] Построение FAISS-индекса      → faiss_index.bin
[4/4] Запуск дашборда на :8501      → Streamlit
```

**Открой http://localhost:8501.**

**Остановить:**

```bash
docker-compose down
```

### 4. Запуск без Docker (альтернатива)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python src/preprocess.py
python src/run_all.py
python src/build_rag_index.py

streamlit run app/dashboard.py
```

## Примеры использования

### Анализ одного текста

```bash
cd ~/nps-analyzer
source venv/bin/activate
python -c "from src.llm_extract import extract; print(extract('Оператору уже 3 года не можем дозвониться'))"
```

**Что вернёт:**

```python
{
  'is_meaningful': True,
  'sentiment': 'negative',
  'sentiment_confidence': 0.95,
  'emotions': ['разочарование', 'раздражение'],
  'entities': {'продукт': ['оператор'], 'проблема': ['не можем дозвониться'], ...},
  'topic': 'обслуживание и операторы',
  'summary': 'Клиент не может дозвониться до оператора 3 года'
}
```

*Требуется запущенная Ollama с моделью `qwen3:8b` на `localhost:11434` (при запуске вне Docker).*

### Поиск похожих отзывов (RAG)

```bash
python -c "from src.rag_search import find_similar; print(find_similar('Дорогой тариф', k=5)[['text', 'sentiment', 'score']])"
```

**Что вернёт:** таблицу из 5 похожих отзывов с оценкой близости (score).

### Статистика по обработанным данным

```bash
python -c "import pandas as pd; df = pd.read_parquet('data/processed/llm_results.parquet'); print(df['sentiment'].value_counts()); print(); print(df['topic'].value_counts().head())"
```

**Что вернёт:**

```
neutral     758
negative    310
positive    305

прочее                    515
связь и покрытие          364
тарифы и оплата           288
обслуживание и операторы  146
```

### Анализ через дашборд

После запуска (`docker-compose up`) откройте **http://localhost:8501**:

1. **Метрики** — 4 ключевых показателя сверху
2. **Тональность** — круговая диаграмма
3. **Темы, эмоции, проблемы** — bar charts
4. **RAG-поиск** — введите запрос («дорогой тариф», «не работает интернет»), получите 5 похожих отзывов с оценкой тональности

## Результаты

На датасете из 1347 анкет:

| Метрика | Значение |
|---|---|
| Всего анкет | 1347 |
| Обработано LLM | 1334 |
| Успешных обработок | 100% |
| Скорость обработки | ~2.2 сек/текст |
| Общее время пайплайна | ~55 минут |
| Тональность | 54% neutral / 25% negative / 21% positive |
| Топ-тема | связь и покрытие |
| Топ-эмоции | нейтрально, разочарование, благодарность |

## LLM Observability (Langfuse)

Все запросы к LLM трейсятся в Langfuse:

- **Latency** — время обработки каждого запроса (~2.2 сек)
- **Tokens** — input / output / total
- **Input / Output** — текст отзыва и структурированный JSON-результат
- **Metadata** — `row_id`, модель, timestamp

Трейсы доступны в [cloud.langfuse.com](https://cloud.langfuse.com) → Traces. Скриншот — в `reports/screenshots/`.


## Пример входных данных

Файл `data/raw/result_nps.xlsx` — выгрузка NPS-опроса. Пример строк:

| create_date | mark_2 | mark_Summ |
|---|---|---|
| 2026-06-24 | Спасибо, всё отлично | Спасибо, всё отлично |
| 2026-07-02 | Оператору уже 3 года не можем дозвониться | Оператору уже 3 года не можем дозвониться |
| 2026-06-15 | Дорого, снизьте тариф | Дорого, снизьте тариф |

## Docker

Единый образ `nps-analyzer:latest` содержит всё: код пайплайна, дашборд, зависимости и данные. При запуске через `entrypoint.sh`:

1. Выполняется пайплайн (preprocess → run_all → build_rag_index).
2. После завершения автоматически запускается Streamlit-дашборд.

Контейнер обращается к Ollama **на хосте** через `host.docker.internal:11434` — GPU-инференс идёт на хосте, а не внутри Docker.

## Видео-демонстрация

Ссылка на видео-демо: https://disk.yandex.ru/i/Fce53aFKz4HJZg

## Лицензия

Учебный проект. Права на исходные данные принадлежат их владельцам.

## Автор

Шульга Н.· 2026
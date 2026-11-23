import datetime as dt
import json
import re
import time
from typing import Literal

import feedparser
from dateutil import parser as date_parser
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field

load_dotenv()

FEEDS: dict[str, list[str]] = {
    "RU": [
        "https://rssexport.rbc.ru/rbcnews/news/30/full.rss",
        "https://www.kommersant.ru/RSS/news.xml",
        "https://www.vedomosti.ru/rss/news",
    ],
    "EU": [
        "https://www.politico.eu/feed/",
        "https://www.euronews.com/rss",
        "https://rss.dw.com/rdf/rss-en-all",
        "https://www.theguardian.com/world/rss",
    ],
    "US": [
        "https://www.federalreserve.gov/feeds/press_all.xml",
        "https://feeds.content.dowjones.io/public/rss/RSSMarketsMain",
        "https://api.axios.com/feed/",
        "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
        "https://techcrunch.com/feed/",
    ],
    "CN": [
        "https://www.scmp.com/rss/91/feed/",
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://www.aljazeera.com/xml/rss/all.xml",
    ],
}

MAX_ITEMS_PER_REGION = 45
DAYS_WINDOW = 7
MODELS_TO_TRY = ["gemini-3.8-flash"]


class NewsItem(BaseModel):
    id: str
    topic: Literal["Политика", "Экономика", "Общество", "Наука"]
    title: str
    summary: str
    impact_country_and_people: str
    importance: Literal["high", "medium"]
    sources: list[str] = Field(default_factory=list)


class RegionsDigest(BaseModel):
    RU: list[NewsItem]
    EU: list[NewsItem]
    US: list[NewsItem]
    CN: list[NewsItem]


class WeeklyDigest(BaseModel):
    week_id: str
    period: str
    generated_at: str
    global_summary: str
    regions: RegionsDigest


def clean_html(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text or "")).strip()


def parse_entry_date(entry) -> dt.datetime | None:
    raw = entry.get("published") or entry.get("updated")
    if not raw:
        return None
    try:
        parsed = date_parser.parse(raw)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc)
    except (ValueError, TypeError):
        return None


def fetch_region_news(region: str, urls: list[str], cutoff: dt.datetime) -> list[dict]:
    collected: list[dict] = []
    seen_links: set[str] = set()

    for url in urls:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            link = (entry.get("link") or "").strip()
            if not link or link in seen_links:
                continue

            pub_dt = parse_entry_date(entry)
            if pub_dt and pub_dt < cutoff:
                continue

            seen_links.add(link)
            title = clean_html(entry.get("title", ""))
            summary = clean_html(entry.get("summary") or entry.get("description", ""))[:500]
            if not title:
                continue

            collected.append(
                {
                    "region": region,
                    "title": title,
                    "summary": summary,
                    "link": link,
                    "published": pub_dt.isoformat() if pub_dt else None,
                }
            )
            if len(collected) >= MAX_ITEMS_PER_REGION:
                return collected

    return collected


def build_prompt(now_utc: dt.datetime, start_dt: dt.datetime, payload_by_region: dict) -> str:
    week_id = f"{now_utc.isocalendar().year}-W{now_utc.isocalendar().week:02d}"
    period = f"{start_dt.strftime('%d.%m.%Y')} – {now_utc.strftime('%d.%m.%Y')}"
    generated_at = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    return f"""
Ты — аналитический редактор еженедельного дайджеста мировых событий.
Сформируй сводку за неделю по четырём регионам: RU, EU, US, CN.

Метаданные выпуска (подставь в соответствующие поля):
- week_id: "{week_id}"
- period: "{period}"
- generated_at: "{generated_at}"

Задача:
1. Из входных новостей отбери только системные и значимые события (политика, экономика, общество, наука/технологии). Отбрось мелкий шум, рекламу и кликбейт.
2. Объединяй дублирующиеся новости об одном сюжете в одну карточку и указывай их ссылки в "sources".
3. Для каждого региона выбери 3–6 наиболее важных событий с балансом тем.
4. Для каждой карточки заполни:
   - "id": короткий уникальный идентификатор (например, "ru-1", "us-2");
   - "topic": строго одно из значений "Политика", "Экономика", "Общество", "Наука";
   - "title": нейтральный заголовок до 10 слов;
   - "summary": суть события (2–3 предложения, факты и цифры);
   - "impact_country_and_people": практический смысл для страны и обычных людей (экономика, доходы, регулирование, сервисы, технологии, международные связи) без общих клише;
   - "importance": "high" или "medium";
   - "sources": список релевантных URL из входных данных.
5. Заполни "global_summary" (4–5 предложений) — главный смысл прошедшей недели в мире.

Входные новости по регионам:
{json.dumps(payload_by_region, ensure_ascii=False, indent=2)}
""".strip()


def call_with_fallback(client: genai.Client, prompt: str) -> WeeklyDigest:
    last_exc: Exception | None = None
    # При пиковой нагрузке или квоте делаем повторные попытки и переключаемся на резервную модель
    for model_name in MODELS_TO_TRY:
        for attempt in range(5):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=WeeklyDigest,
                        temperature=0.2,
                    ),
                )
                return WeeklyDigest.model_validate_json(response.text)
            except errors.APIError as exc:
                last_exc = exc
                if exc.code in (429, 500, 503, 504):
                    time.sleep(5 * (attempt + 1))
                    continue
                raise
    raise RuntimeError(f"Не удалось получить ответ от API: {last_exc}")


def filter_valid_sources(digest: WeeklyDigest, raw_by_region: dict[str, list[dict]]) -> WeeklyDigest:
    allowed_links = {
        item["link"]
        for items in raw_by_region.values()
        for item in items
        if item.get("link")
    }
    # Оставляем только те URL, которые реально пришли из RSS-лент на текущем запуске
    for region_items in (
        digest.regions.RU,
        digest.regions.EU,
        digest.regions.US,
        digest.regions.CN,
    ):
        for card in region_items:
            card.sources = [url for url in card.sources if url in allowed_links]
    return digest


def generate_digest(output_path: str = "weekly_digest.json") -> None:
    now_utc = dt.datetime.now(dt.timezone.utc)
    cutoff = now_utc - dt.timedelta(days=DAYS_WINDOW)

    raw_by_region = {
        region: fetch_region_news(region, urls, cutoff)
        for region, urls in FEEDS.items()
    }

    client = genai.Client()
    prompt = build_prompt(now_utc, cutoff, raw_by_region)
    digest = call_with_fallback(client, prompt)
    digest = filter_valid_sources(digest, raw_by_region)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(digest.model_dump_json(indent=2))

    print(f"Готово: {output_path} ({digest.week_id})")


if __name__ == "__main__":
    generate_digest()

"""
NLP Sentiment Engine — Scrapes PSX-relevant headlines and scores them via VADER.
Returns a live compound sentiment score (-1.0 to +1.0).
"""
import logging
import time
import requests
import threading
from bs4 import BeautifulSoup

logger = logging.getLogger('nexus_ai')

# ---------------------------------------------------------------------------
# VADER — lazy-loaded so nltk data is downloaded only once
# ---------------------------------------------------------------------------
_vader = None
_vader_lock = threading.Lock()

def _get_vader():
    global _vader
    if _vader is None:
        with _vader_lock:
            if _vader is None:
                import nltk
                try:
                    from nltk.sentiment.vader import SentimentIntensityAnalyzer
                    _vader = SentimentIntensityAnalyzer()
                except LookupError:
                    nltk.download('vader_lexicon', quiet=True)
                    from nltk.sentiment.vader import SentimentIntensityAnalyzer
                    _vader = SentimentIntensityAnalyzer()
                logger.info("VADER SentimentIntensityAnalyzer initialised.")
    return _vader


# ---------------------------------------------------------------------------
# News Scraping
# ---------------------------------------------------------------------------
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

_SOURCES = [
    {
        "name": "Dawn Business",
        "url": "https://www.dawn.com/business",
        "selector": "article h2 a, .story__title a",
    },
    {
        "name": "Express Tribune Market",
        "url": "https://tribune.com.pk/business/market",
        "selector": "h2 a, .story-title a, .article-title a",
    },
]

# Google News RSS — Cloudflare-proof fallback
_RSS_QUERIES = [
    "PSX+stock+market+Pakistan",
    "Pakistan+stock+exchange+OGDC",
]


def _scrape_google_news_rss(timeout=8):
    """Fetch PSX-related headlines from Google News RSS (no Cloudflare)."""
    headlines = []
    for query in _RSS_QUERIES:
        url = f"https://news.google.com/rss/search?q={query}&hl=en-PK&gl=PK&ceid=PK:en"
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=timeout)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.content, "xml")
            items = soup.find_all("item")
            for item in items:
                title = item.find("title")
                if title:
                    text = title.get_text(strip=True)
                    if text and len(text) > 10:
                        headlines.append(text)
        except Exception as e:
            logger.warning(f"Google News RSS failed for '{query}': {e}")
    
    # Deduplicate while preserving order
    headlines = list(dict.fromkeys(headlines))

    if headlines:
        logger.info(f"Google News RSS: scraped {len(headlines)} headlines")
    return headlines


def scrape_psx_headlines(timeout=8):
    """Scrape business headlines from Pakistani news sites.
    Falls back to Google News RSS if direct scraping is Cloudflare-blocked.
    Returns a list of headline strings.
    """
    headlines = []
    for source in _SOURCES:
        try:
            resp = requests.get(
                source["url"], headers=_HEADERS, timeout=timeout
            )
            if resp.status_code != 200:
                logger.warning(f"{source['name']}: HTTP {resp.status_code}")
                continue
            soup = BeautifulSoup(resp.content, "html.parser")
            tags = soup.select(source["selector"])
            for tag in tags:
                text = tag.get_text(strip=True)
                if text and len(text) > 10:  # skip tiny fragments
                    headlines.append(text)
            logger.info(f"{source['name']}: scraped {len(tags)} headlines")
        except Exception as e:
            logger.warning(f"{source['name']} scrape failed: {e}")

    # Fallback: Google News RSS if direct scraping yielded nothing
    if not headlines:
        logger.info("Direct scraping returned 0 headlines — falling back to Google News RSS")
        headlines = _scrape_google_news_rss(timeout)

    return headlines


# ---------------------------------------------------------------------------
# Sentiment Analysis
# ---------------------------------------------------------------------------
def analyze_sentiment(headlines):
    """Run VADER on each headline, return mean compound score.
    Score range: -1.0 (very bearish) to +1.0 (very bullish).
    """
    if not headlines:
        return 0.0
    vader = _get_vader()
    compounds = [vader.polarity_scores(h)["compound"] for h in headlines]
    mean_score = sum(compounds) / len(compounds)
    return round(mean_score, 4)


# ---------------------------------------------------------------------------
# Cached Orchestrator
# ---------------------------------------------------------------------------
# Cached Orchestrator
# ---------------------------------------------------------------------------
_cache = {"score": 0.0, "label": "Neutral", "count": 0, "ts": 0}
_CACHE_TTL = 900  # 15 minutes
_cache_lock = threading.Lock()


def get_live_sentiment():
    """Scrape → Analyse → Return cached result.
    Returns dict: {"score": float, "label": str, "headline_count": int}
    """
    now = time.time()
    
    # 1. Fast path: Read cache without lock
    if now - _cache["ts"] < _CACHE_TTL and _cache["ts"] > 0:
        return {
            "score": _cache["score"],
            "label": _cache["label"],
            "headline_count": _cache["count"],
        }

    # 2. Slow path: Cache expired, acquire lock
    with _cache_lock:
        # 3. Double-check timestamp inside lock (in case another thread updated it)
        if now - _cache["ts"] < _CACHE_TTL and _cache["ts"] > 0:
            return {
                "score": _cache["score"],
                "label": _cache["label"],
                "headline_count": _cache["count"],
            }

        try:
            headlines = scrape_psx_headlines()
            score = analyze_sentiment(headlines)

            if score >= 0.15:
                label = "Bullish"
            elif score <= -0.15:
                label = "Bearish"
            else:
                label = "Neutral"

            _cache.update({"score": score, "label": label,
                            "count": len(headlines), "ts": now})
            logger.info(
                f"Sentiment updated: {score:.4f} ({label}) "
                f"from {len(headlines)} headlines"
            )
        except Exception as e:
            logger.error(f"Sentiment pipeline failed: {e}")

    return {
        "score": _cache["score"],
        "label": _cache["label"],
        "headline_count": _cache["count"],
    }

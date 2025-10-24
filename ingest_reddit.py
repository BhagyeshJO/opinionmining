import os, time, json, requests, sys, hashlib
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin, quote_plus

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
# Allow a single keyword or a pipe-separated set of phrases:
# e.g., "Should you buy Silver|silver investment|buy silver now"
RAW_KW = os.environ.get("KEYWORD", "Should you buy Silver")
QUERIES = [q.strip() for q in RAW_KW.split("|") if q.strip()]
HOURS = int(os.environ.get("LOOKBACK_HOURS", "24"))
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

if not all([SUPABASE_URL, SUPABASE_ANON_KEY, HF_TOKEN]):
    print("✗ Missing one of SUPABASE_URL / SUPABASE_ANON_KEY / HF_TOKEN", file=sys.stderr)
    sys.exit(1)

SINCE_DT = datetime.now(timezone.utc) - timedelta(hours=HOURS)
SINCE_UNIX = int(SINCE_DT.timestamp())

HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}
SB_HEADERS = {
    "apikey": SUPABASE_ANON_KEY,
    "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
    "Content-Type": "application/json",
    "Prefer": "resolution=merge-duplicates",
}
UA = {"User-Agent": "opinion-miner/0.1 by github-actions"}

def reddit_search(query, limit=50, t="week"):
    # Reddit public search endpoint (no auth). Returns posts, not comments.
    url = "https://www.reddit.com/search.json"
    try:
        r = requests.get(
            url,
            params={"q": query, "sort": "new", "limit": str(limit), "t": t, "type": "link"},
            headers=UA,
            timeout=30
        )
        r.raise_for_status()
        data = r.json().get("data", {}).get("children", [])
        posts = []
        for item in data:
            d = item.get("data", {})
            # Filter by created_utc >= SINCE
            created = d.get("created_utc")
            if created and int(created) < SINCE_UNIX:
                continue
            posts.append({
                "id": d.get("id"),
                "url": f"https://www.reddit.com{d.get('permalink','')}",
                "author": d.get("author"),
                "title": d.get("title") or "",
                "selftext": d.get("selftext") or "",
                "subreddit": d.get("subreddit"),
                "created_utc": created or 0
            })
        print(f"✓ Reddit search '{query}' → {len(posts)} posts")
        return posts
    except Exception as e:
        print(f"⚠️ Reddit search failed for '{query}': {e}", file=sys.stderr)
        return []

def hf_sentiment(text, retries=3):
    text = (text or "").strip()
    if not text:
        return "neutral", 0.0
    payload = {"inputs": text[:5000]}
    url = f"https://api-inference.huggingface.co/models/{MODEL}"
    for i in range(retries):
        try:
            r = requests.post(url, headers=HF_HEADERS, json=payload, timeout=60)
            if r.status_code == 503:
                print("… HF 503 (loading), retrying in 6s")
                time.sleep(6)
                continue
            r.raise_for_status()
            out = r.json()
            if isinstance(out, list) and out and isinstance(out[0], list):
                out = out[0]
            best = max(out, key=lambda x: x.get("score", 0)) if isinstance(out, list) else {"label":"neutral","score":0.0}
            return best["label"].lower(), float(best["score"])
        except Exception as e:
            print(f"⚠️ HF call failed (attempt {i+1}/{retries}): {e}", file=sys.stderr)
            time.sleep(3)
    return "neutral", 0.0

def upsert_post(p):
    url = urljoin(SUPABASE_URL, "/rest/v1/posts")
    r = requests.post(url, headers=SB_HEADERS, data=json.dumps([p]), timeout=60)
    r.raise_for_status()
    return r.json()[0]["id"]

def insert_analysis(a):
    url = urljoin(SUPABASE_URL, "/rest/v1/analysis")
    r = requests.post(url, headers=SB_HEADERS, data=json.dumps([a]), timeout=60)
    r.raise_for_status()

def main():
    all_posts = []
    for q in QUERIES:
        all_posts.extend(reddit_search(q, limit=50, t="month"))

    if not all_posts:
        print("No posts fetched; try broadening KEYWORD or increasing LOOKBACK_HOURS.")
        return

    inserted, analyzed = 0, 0
    for p in all_posts:
        text_block = (p["title"] + "\n\n" + p["selftext"]).strip()
        if not text_block:
            continue

        # Create a stable external_id based on reddit id + URL (for de-dupe safety)
        ext = p["id"] or p["url"]
        if not ext:
            ext = hashlib.sha256(text_block.encode("utf-8")).hexdigest()[:16]

        post = {
            "source": "reddit",
            "external_id": str(ext),
            "url": p["url"],
            "author": p["author"],
            "title": p["title"],
            "text": text_block,
            "language": "en",
        }
        try:
            post_id = upsert_post(post)
            inserted += 1
        except Exception as e:
            print(f"⚠️ Post upsert failed for {ext}: {e}", file=sys.stderr)
            continue

        label, score = hf_sentiment(text_block)
        a = {
            "post_id": post_id,
            "sentiment": label,
            "sentiment_score": round(score, 4),
            "model": MODEL,
        }
        try:
            insert_analysis(a)
            analyzed += 1
        except Exception as e:
            print(f"⚠️ Analysis insert failed for {post_id}: {e}", file=sys.stderr)
        time.sleep(0.4)

    print(f"✓ Done. Upserted posts: {inserted}, analyses saved: {analyzed}")

if __name__ == "__main__":
    main()

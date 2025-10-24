import os, time, json, requests, sys
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
KEYWORD = os.environ.get("KEYWORD", "Should you buy Silver")
HOURS = int(os.environ.get("LOOKBACK_HOURS", "6"))
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

if not all([SUPABASE_URL, SUPABASE_ANON_KEY, HF_TOKEN]):
    print("✗ Missing one of SUPABASE_URL / SUPABASE_ANON_KEY / HF_TOKEN", file=sys.stderr)
    sys.exit(1)

SINCE = int((datetime.now(timezone.utc) - timedelta(hours=HOURS)).timestamp())
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}
SB_HEADERS = {
    "apikey": SUPABASE_ANON_KEY,
    "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
    "Content-Type": "application/json",
    "Prefer": "resolution=merge-duplicates",
}

def fetch_reddit(keyword, since):
    url = "https://api.pushshift.io/reddit/search/comment/"
    try:
        r = requests.get(url, params={"q": keyword, "after": since, "size": 50}, timeout=30)
        r.raise_for_status()
        data = r.json().get("data", [])
        print(f"✓ Fetched {len(data)} comments for '{keyword}' since {since}")
        return data
    except Exception as e:
        print(f"⚠️ Reddit fetch failed: {e}", file=sys.stderr)
        return []

def hf_sentiment(text, retries=3):
    payload = {"inputs": text[:5000]}
    url = f"https://api-inference.huggingface.co/models/{MODEL}"
    for i in range(retries):
        try:
            r = requests.post(url, headers=HF_HEADERS, json=payload, timeout=60)
            if r.status_code == 503:
                # model cold-start
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
    comments = fetch_reddit(KEYWORD, SINCE)
    if not comments:
        print("No comments fetched; exiting gracefully.")
        return
    inserted, analyzed = 0, 0
    for c in comments:
        text = c.get("body") or ""
        if not text.strip():
            continue
        post = {
            "source": "reddit",
            "external_id": str(c.get("id")),
            "url": f"https://reddit.com{c.get('permalink','')}",
            "author": c.get("author"),
            "title": None,
            "text": text,
            "language": c.get("lang") or "en",
        }
        try:
            post_id = upsert_post(post)
            inserted += 1
        except Exception as e:
            # unique conflict is handled by Prefer=merge-duplicates, but still guard
            print(f"⚠️ Post upsert failed for {post.get('external_id')}: {e}", file=sys.stderr)
            continue

        label, score = hf_sentiment(text)
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
        time.sleep(0.4)  # rate friendly
    print(f"✓ Done. Upserted posts: {inserted}, analyses saved: {analyzed}")

if __name__ == "__main__":
    main()

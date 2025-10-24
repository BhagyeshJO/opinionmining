import os, time, json, requests, sys, hashlib
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin

# ---------- Config ----------
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
RAW_KW = os.environ.get("KEYWORD", "Should you buy Silver")
QUERIES = [q.strip() for q in RAW_KW.split("|") if q.strip()]
HOURS = int(os.environ.get("LOOKBACK_HOURS", "168"))  # 7 days first run
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET")
REDDIT_USERNAME = os.environ.get("REDDIT_USERNAME")  # real username, not display name
REDDIT_PASSWORD = os.environ.get("REDDIT_PASSWORD")

if not all([SUPABASE_URL, SUPABASE_ANON_KEY, HF_TOKEN, REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET]):
    print("✗ Missing required env vars. Need SUPABASE_URL, SUPABASE_ANON_KEY, HF_TOKEN, REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET", file=sys.stderr)
    sys.exit(1)

UA_STR = "opinion-miner/0.2 (by u/{})".format(REDDIT_USERNAME or "github-actions")
UA = {"User-Agent": UA_STR}
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}
SB_HEADERS = {
    "apikey": SUPABASE_ANON_KEY,
    "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
    "Content-Type": "application/json",
    "Prefer": "resolution=merge-duplicates",
}
SINCE_UNIX = int((datetime.now(timezone.utc) - timedelta(hours=HOURS)).timestamp())

# ---------- Reddit OAuth helpers ----------
def reddit_token_password():
    """Password grant (requires script app + username/password)."""
    auth = requests.auth.HTTPBasicAuth(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET)
    data = {
        "grant_type": "password",
        "username": REDDIT_USERNAME,
        "password": REDDIT_PASSWORD,
        "scope": "read",
    }
    r = requests.post("https://www.reddit.com/api/v1/access_token",
                      auth=auth, data=data, headers=UA, timeout=30)
    if r.status_code != 200:
        print(f"⚠️ Password grant failed ({r.status_code}): {r.text}", file=sys.stderr)
        return None
    try:
        j = r.json()
    except Exception:
        print(f"⚠️ Password grant returned non-JSON: {r.text}", file=sys.stderr)
        return None
    tok = j.get("access_token")
    if not tok:
        print(f"⚠️ Password grant missing access_token: {j}", file=sys.stderr)
        return None
    return tok

def reddit_token_app_only():
    """Application-only (client_credentials). No username/password."""
    auth = requests.auth.HTTPBasicAuth(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET)
    data = {"grant_type": "client_credentials", "scope": "read"}
    r = requests.post("https://www.reddit.com/api/v1/access_token",
                      auth=auth, data=data, headers=UA, timeout=30)
    if r.status_code != 200:
        print(f"⚠️ Client credentials failed ({r.status_code}): {r.text}", file=sys.stderr)
        return None
    j = r.json()
    tok = j.get("access_token")
    if not tok:
        print(f"⚠️ Client credentials missing access_token: {j}", file=sys.stderr)
        return None
    return tok

def get_reddit_token():
    # Try password grant first if creds present
    if REDDIT_USERNAME and REDDIT_PASSWORD:
        tok = reddit_token_password()
        if tok:
            print("✓ Reddit token via password grant")
            return tok
        print("⚠️ Falling back to application-only token...")
    # Fallback: app-only
    tok = reddit_token_app_only()
    if tok:
        print("✓ Reddit token via client_credentials")
        return tok
    print("✗ Reddit auth failed (both grants).", file=sys.stderr)
    sys.exit(1)

def reddit_search_bearer(bearer, query, limit=50, sort="new", timeframe="year"):
    headers = {"Authorization": f"bearer {bearer}", **UA}
    params = {"q": query, "limit": str(limit), "sort": sort, "t": timeframe, "type": "link", "restrict_sr": False}
    r = requests.get("https://oauth.reddit.com/search", headers=headers, params=params, timeout=30)
    if r.status_code != 200:
        raise requests.HTTPError(f"{r.status_code}: {r.text}")
    data = r.json().get("data", {}).get("children", [])
    posts = []
    for item in data:
        d = item.get("data", {})
        created = int(d.get("created_utc") or 0)
        if created and created < SINCE_UNIX:
            continue
        posts.append({
            "id": d.get("id"),
            "url": f"https://www.reddit.com{d.get('permalink','')}",
            "author": d.get("author"),
            "title": d.get("title") or "",
            "selftext": d.get("selftext") or "",
            "subreddit": d.get("subreddit"),
            "created_utc": created
        })
    print(f"✓ Reddit search '{query}' → {len(posts)} posts")
    return posts

# ---------- HF, Supabase ----------
def hf_sentiment(text, retries=3):
    text = (text or "").strip()
    if not text:
        return "neutral", 0.0
    payload = {"inputs": text[:5000]}
    url = f"https://api-inference.huggingface.co/models/{MODEL}"
    for i in range(retries):
        r = requests.post(url, headers=HF_HEADERS, json=payload, timeout=60)
        if r.status_code == 503:
            print("… HF 503 (loading), retrying in 6s")
            time.sleep(6); continue
        try:
            r.raise_for_status()
        except Exception as e:
            print(f"⚠️ HF error: {r.status_code} {r.text}", file=sys.stderr)
            time.sleep(3); continue
        out = r.json()
        if isinstance(out, list) and out and isinstance(out[0], list):
            out = out[0]
        best = max(out, key=lambda x: x.get("score", 0)) if isinstance(out, list) else {"label":"neutral","score":0.0}
        return best["label"].lower(), float(best["score"])
    return "neutral", 0.0

def upsert_post(p):
    r = requests.post(urljoin(SUPABASE_URL, "/rest/v1/posts"),
                      headers=SB_HEADERS, data=json.dumps([p]), timeout=60)
    r.raise_for_status()
    return r.json()[0]["id"]

def insert_analysis(a):
    r = requests.post(urljoin(SUPABASE_URL, "/rest/v1/analysis"),
                      headers=SB_HEADERS, data=json.dumps([a]), timeout=60)
    r.raise_for_status()

# ---------- Main ----------
def main():
    bearer = get_reddit_token()

    queries = QUERIES if QUERIES else ["Should you buy Silver"]
    all_posts = []
    for q in queries:
        try:
            all_posts.extend(reddit_search_bearer(bearer, q, limit=50, sort="new", timeframe="year"))
        except Exception as e:
            print(f"⚠️ Reddit search failed for '{q}': {e}", file=sys.stderr)

    if not all_posts:
        print("No posts fetched via Reddit OAuth. Try broadening KEYWORD or increasing LOOKBACK_HOURS.")
        return

    inserted, analyzed = 0, 0
    for p in all_posts:
        text_block = (p["title"] + ("\n\n" + p["selftext"] if p["selftext"] else "")).strip()
        if not text_block:
            continue
        ext = p["id"] or p["url"] or hashlib.sha256(text_block.encode("utf-8")).hexdigest()[:16]
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
            post_id = upsert_post(post); inserted += 1
        except Exception as e:
            print(f"⚠️ Post upsert failed for {ext}: {e}", file=sys.stderr)
            continue

        label, score = hf_sentiment(text_block)
        try:
            insert_analysis({
                "post_id": post_id,
                "sentiment": label,
                "sentiment_score": round(score, 4),
                "model": MODEL,
            }); analyzed += 1
        except Exception as e:
            print(f"⚠️ Analysis insert failed for {post_id}: {e}", file=sys.stderr)
        time.sleep(0.3)

    print(f"✓ Done. Upserted posts: {inserted}, analyses saved: {analyzed}")

if __name__ == "__main__":
    main()

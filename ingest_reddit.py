import os, time, requests, json
KW = os.getenv("KEYWORD", "your brand")
HOURS = int(os.getenv("LOOKBACK_HOURS", "6"))
SINCE = int((datetime.now(timezone.utc) - timedelta(hours=HOURS)).timestamp())


# Simple Reddit via pushshift mirror (community‑run; may be spotty). Prefer official API for reliability.
r = requests.get(
"https://api.pushshift.io/reddit/search/comment/",
params={"q": KW, "after": SINCE, "size": 50}
)
comments = r.json().get("data", [])


# Insert posts & run sentiment via HF Inference API
for c in comments:
text = c.get("body") or ""
if not text.strip():
continue


# Upsert post
post = {
"source": "reddit",
"external_id": str(c.get("id")),
"url": f"https://reddit.com{c.get('permalink','')}",
"author": c.get("author"),
"title": None,
"text": text,
"language": c.get("lang") or "en",
}
from urllib.parse import urljoin
sup_url = urljoin(SUPABASE_URL, "/rest/v1/posts")
h = {"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {SUPABASE_ANON_KEY}", "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates"}
ins = requests.post(sup_url, headers=h, data=json.dumps([post]))
ins.raise_for_status()
post_id = ins.json()[0]["id"]


# Sentiment
resp = requests.post(
f"https://api-inference.huggingface.co/models/{MODEL}",
headers=HEADERS,
json={"inputs": text[:5000]}
)
resp.raise_for_status()
out = resp.json()[0]
# Map to label & score
best = max(out, key=lambda x: x["score"]) if isinstance(out, list) else {"label":"neutral","score":0.0}


# Save analysis
a = {
"post_id": post_id,
"sentiment": best["label"].lower(),
"sentiment_score": round(float(best["score"]), 4),
"model": MODEL
}
sup_a = urljoin(SUPABASE_URL, "/rest/v1/analysis")
ain = requests.post(sup_a, headers=h, data=json.dumps([a]))
ain.raise_for_status()
time.sleep(0.4) # be gentle
print(f"Ingested {len(comments)} comments for '{KW}'.")

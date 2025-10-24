import os, pandas as pd
import streamlit as st
from supabase import create_client
from dateutil import parser

st.set_page_config(page_title="Opinion Miner", layout="wide")
st.title("📊 Social Opinion Miner")

# ---- Supabase client ----
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")
client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# ---- Controls ----
with st.sidebar:
    st.header("Filters")
    src = st.multiselect("Source", ["reddit","youtube","x"], default=["reddit"])
    lookback_days = st.slider("Lookback (days)", 1, 30, 7)
    contains = st.text_input("Text contains (optional)", "")
    st.caption("Tip: try words like *premium*, *manipulation*, *ETF*, *inflation*")

# ---- Load posts (last N days) ----
from datetime import datetime, timedelta, timezone
since = datetime.now(timezone.utc) - timedelta(days=lookback_days)

posts_q = (
    client.table("posts")
    .select("id,source,title,text,url,created_at")
    .order("created_at", desc=True)
    .limit(2000)
)

if src:
    posts_q = posts_q.in_("source", src)

posts = posts_q.execute().data or []
df = pd.DataFrame(posts)

# Filter by time & keyword
if not df.empty:
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df[df["created_at"] >= since]
    if contains:
        mask = (
            df["text"].fillna("").str.contains(contains, case=False, na=False) |
            df["title"].fillna("").str.contains(contains, case=False, na=False)
        )
        df = df[mask]

# ---- Join with analysis ----
if not df.empty:
    ids = df["id"].tolist()
    analysis = client.table("analysis").select(
        "post_id,sentiment,sentiment_score,analyzed_at"
    ).in_("post_id", ids).execute().data or []
    adf = pd.DataFrame(analysis)
    adf["analyzed_at"] = pd.to_datetime(adf["analyzed_at"], utc=True, errors="coerce")
    data = df.merge(adf, left_on="id", right_on="post_id", how="left")
else:
    data = df

# ---- KPIs ----
col1, col2, col3 = st.columns(3)
col1.metric("Posts", len(data))
if "sentiment_score" in data:
    col2.metric("Avg sentiment", round(float(data["sentiment_score"].fillna(0).mean()), 3))
    col3.metric("Positive share", f"{(data['sentiment'].eq('positive').mean()*100):.1f}%")
else:
    col2.metric("Avg sentiment", "-")
    col3.metric("Positive share", "-")

# ---- Charts ----
st.subheader("Sentiment distribution")
if not data.empty and "sentiment" in data:
    st.bar_chart(data["sentiment"].value_counts())

st.subheader("Sentiment over time (daily avg)")
if not data.empty and "created_at" in data and "sentiment_score" in data:
    daily = data.set_index("created_at").resample("D")["sentiment_score"].mean().fillna(0)
    st.line_chart(daily)

# ---- Table of recent posts ----
st.subheader("Recent posts")
cols = ["created_at","source","sentiment","sentiment_score","title","text","url"]
if not data.empty:
    data = data.sort_values("created_at", ascending=False)
    st.dataframe(data[cols], use_container_width=True, height=500)
else:
    st.info("No rows match your filters yet. Try increasing lookback or clearing filters.")

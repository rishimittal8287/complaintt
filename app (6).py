"""
Signal Desk — GenAI Customer Complaint Analytics
--------------------------------------------------
A Streamlit app that logs customer complaints and uses the Claude API
to triage each one: category, sentiment, urgency (1-5), a one-line
summary, and a suggested agent reply. Includes a live dashboard.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Run in Google Colab:
    !pip install -r requirements.txt -q
    !pip install pyngrok -q
    from pyngrok import ngrok
    ngrok.set_auth_token("YOUR_NGROK_TOKEN")   # https://dashboard.ngrok.com
    public_url = ngrok.connect(8501)
    print(public_url)
    !streamlit run app.py &>/content/logs.txt &

You'll need an Anthropic API key: https://console.anthropic.com/
"""

import json
import os
import uuid
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from anthropic import Anthropic

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
MODEL = "claude-sonnet-5"

CATEGORIES = [
    "Billing", "Product Quality", "Shipping & Delivery",
    "Customer Service", "Technical Issue", "Refund/Return", "Other",
]
CHANNELS = ["Email", "Phone", "Chat", "Social Media", "In-Store"]

CATEGORY_COLOR = {
    "Billing": "#f59e0b", "Product Quality": "#f43f5e",
    "Shipping & Delivery": "#38bdf8", "Customer Service": "#a78bfa",
    "Technical Issue": "#fb923c", "Refund/Return": "#2dd4bf", "Other": "#94a3b8",
}
SENTIMENT_COLOR = {"negative": "#f43f5e", "neutral": "#94a3b8", "positive": "#2dd4bf"}

SAMPLE_COMPLAINTS = [
    {"customer": "Priya Nair", "channel": "Email", "text": "My order #48213 was supposed to arrive 5 days ago. Tracking hasn't updated since Tuesday and support hasn't replied to two emails. I need this resolved today or I want a full refund."},
    {"customer": "Daniel Cho", "channel": "Chat", "text": "The blender I bought stopped working after two uses. The blades won't even spin. For the price I paid I expected much better build quality."},
    {"customer": "Amara Obi", "channel": "Phone", "text": "I was charged twice for my last subscription renewal. I can see both charges on my statement. Please refund the duplicate charge as soon as possible."},
    {"customer": "Liam Fischer", "channel": "Social Media", "text": "Third time asking about this and still no response from your team. Genuinely considering cancelling everything after this experience."},
    {"customer": "Sofia Marquez", "channel": "In-Store", "text": "The staff at the downtown location were dismissive when I asked about a return. Not the experience I expect from this brand."},
    {"customer": "Noah Becker", "channel": "Email", "text": "App keeps crashing every time I try to check out on iOS. Lost my cart contents twice now. Can you look into this bug please?"},
    {"customer": "Grace Lin", "channel": "Chat", "text": "Just wanted to say the replacement part arrived quickly and the instructions were clear. Fixed the issue myself in ten minutes, thank you."},
]

SYSTEM_PROMPT = f"""You are a customer complaint triage assistant. Given a single customer \
complaint, respond with ONLY a raw JSON object, no markdown fences, no preamble, matching \
exactly this shape:
{{"category": one of {CATEGORIES}, "sentiment": one of ["negative","neutral","positive"], \
"urgency": integer 1-5 (5 = most urgent, e.g. threat to cancel, safety issue, repeated \
unresolved contact), "summary": a plain one-sentence summary under 18 words, \
"suggestedResponse": a short, empathetic two-sentence reply a support agent could send}}"""

# ----------------------------------------------------------------------
# State
# ----------------------------------------------------------------------
st.set_page_config(page_title="Signal Desk", page_icon="📡", layout="wide")

if "tickets" not in st.session_state:
    st.session_state.tickets = []  # list of dicts


def new_ticket(customer, channel, text):
    return {
        "id": str(uuid.uuid4()),
        "customer": customer or "Anonymous",
        "channel": channel,
        "text": text,
        "created_at": datetime.now(),
        "status": "pending",  # pending | done | error
        "analysis": None,
    }


def analyze_complaint(client, text):
    resp = client.messages.create(
        model=MODEL,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text}],
    )
    raw = "".join(block.text for block in resp.content if block.type == "text").strip()
    cleaned = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(cleaned)


# ----------------------------------------------------------------------
# Sidebar — API key + intake form
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📡 Signal Desk")
    st.caption("GenAI customer complaint triage & analytics")

    api_key = st.text_input(
        "Anthropic API key",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Get one at https://console.anthropic.com/",
    )

    st.divider()
    st.markdown("**Log a complaint**")
    with st.form("intake_form", clear_on_submit=True):
        customer = st.text_input("Customer name")
        channel = st.selectbox("Channel", CHANNELS)
        text = st.text_area("Complaint text", height=120)
        submitted = st.form_submit_button("➕ Add to queue", use_container_width=True)
        if submitted and text.strip():
            st.session_state.tickets.insert(0, new_ticket(customer, channel, text.strip()))

    if st.button("🔄 Load sample complaints", use_container_width=True):
        st.session_state.tickets = [new_ticket(**c) for c in SAMPLE_COMPLAINTS] + st.session_state.tickets

    if st.session_state.tickets and st.button("🗑️ Clear queue", use_container_width=True):
        st.session_state.tickets = []
        st.rerun()

# ----------------------------------------------------------------------
# Main — header + stats
# ----------------------------------------------------------------------
st.title("Signal Desk")
st.caption("Log incoming complaints, run them through AI triage, and watch category, sentiment and urgency patterns surface as they arrive.")

tickets = st.session_state.tickets
analyzed = [t for t in tickets if t["status"] == "done"]
pending = [t for t in tickets if t["status"] in ("pending", "error")]

col1, col2, col3, col4 = st.columns(4)
col1.metric("In queue", len(tickets), f"{len(analyzed)} analyzed")
avg_urgency = round(sum(t["analysis"]["urgency"] for t in analyzed) / len(analyzed), 1) if analyzed else "–"
col2.metric("Avg urgency", avg_urgency, help="Scale of 1-5")
critical = sum(1 for t in analyzed if t["analysis"]["urgency"] >= 4)
col3.metric("Critical (4-5)", critical)
if analyzed:
    cat_counts = pd.Series([t["analysis"]["category"] for t in analyzed]).value_counts()
    top_category = cat_counts.idxmax()
else:
    top_category = "–"
col4.metric("Top category", top_category)

if pending:
    if st.button(f"✨ Analyze all pending ({len(pending)})", type="primary"):
        if not api_key:
            st.error("Add your Anthropic API key in the sidebar first.")
        else:
            client = Anthropic(api_key=api_key)
            progress = st.progress(0.0, text="Running AI triage...")
            for i, t in enumerate(pending):
                try:
                    t["analysis"] = analyze_complaint(client, t["text"])
                    t["status"] = "done"
                except Exception as e:
                    t["status"] = "error"
                    st.warning(f"Failed to analyze complaint from {t['customer']}: {e}")
                progress.progress((i + 1) / len(pending))
            progress.empty()
            st.rerun()

st.divider()

left, right = st.columns([2, 3])

# ----------------------------------------------------------------------
# Left — queue
# ----------------------------------------------------------------------
with left:
    st.markdown("#### Queue")
    if not tickets:
        st.info("Queue is empty. Add a complaint or load samples from the sidebar.")
    for t in tickets:
        with st.container(border=True):
            header = f"**{t['customer']}** · {t['channel']}"
            if t["analysis"]:
                header += f" · :{'red' if t['analysis']['urgency'] >= 4 else 'orange' if t['analysis']['urgency'] >= 2 else 'green'}[{t['analysis']['category']}] · urgency {t['analysis']['urgency']}/5"
            st.markdown(header)
            st.caption(t["text"][:220] + ("..." if len(t["text"]) > 220 else ""))

            if t["status"] == "done":
                with st.expander("AI triage details"):
                    a = t["analysis"]
                    st.markdown(f"**Summary:** {a['summary']}")
                    st.markdown(f"**Sentiment:** {a['sentiment'].capitalize()}")
                    st.markdown(f"**Suggested reply:** _{a['suggestedResponse']}_")
            else:
                bcol1, bcol2 = st.columns([3, 1])
                if bcol1.button("Run AI triage" if t["status"] == "pending" else "Retry analysis",
                                 key=f"run_{t['id']}"):
                    if not api_key:
                        st.error("Add your Anthropic API key in the sidebar first.")
                    else:
                        client = Anthropic(api_key=api_key)
                        try:
                            t["analysis"] = analyze_complaint(client, t["text"])
                            t["status"] = "done"
                        except Exception as e:
                            t["status"] = "error"
                            st.error(f"Analysis failed: {e}")
                        st.rerun()
                if bcol2.button("🗑️", key=f"del_{t['id']}"):
                    st.session_state.tickets = [x for x in st.session_state.tickets if x["id"] != t["id"]]
                    st.rerun()

# ----------------------------------------------------------------------
# Right — analytics
# ----------------------------------------------------------------------
with right:
    st.markdown("#### Complaints by category")
    if analyzed:
        df_cat = pd.Series([t["analysis"]["category"] for t in analyzed]).value_counts().reset_index()
        df_cat.columns = ["category", "count"]
        fig = px.bar(
            df_cat, x="count", y="category", orientation="h",
            color="category", color_discrete_map=CATEGORY_COLOR,
        )
        fig.update_layout(showlegend=False, height=280, margin=dict(l=0, r=0, t=10, b=0),
                           yaxis_title=None, xaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No analyzed complaints yet.")

    st.markdown("#### Sentiment split")
    if analyzed:
        df_sent = pd.Series([t["analysis"]["sentiment"] for t in analyzed]).value_counts().reset_index()
        df_sent.columns = ["sentiment", "count"]
        fig2 = px.pie(
            df_sent, names="sentiment", values="count", hole=0.55,
            color="sentiment", color_discrete_map=SENTIMENT_COLOR,
        )
        fig2.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.caption("No analyzed complaints yet.")

    if analyzed:
        st.markdown("#### All analyzed complaints")
        df_all = pd.DataFrame([{
            "Customer": t["customer"],
            "Channel": t["channel"],
            "Category": t["analysis"]["category"],
            "Sentiment": t["analysis"]["sentiment"],
            "Urgency": t["analysis"]["urgency"],
            "Summary": t["analysis"]["summary"],
        } for t in analyzed])
        st.dataframe(df_all, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download as CSV",
            df_all.to_csv(index=False).encode("utf-8"),
            file_name="complaint_analysis.csv",
            mime="text/csv",
        )

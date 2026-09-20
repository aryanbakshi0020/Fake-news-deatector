import streamlit as st

from detector import DATASET_PATH, explain_prediction, load_and_prepare_data, predict_news, train_model

st.set_page_config(page_title="TruthLens | News Intelligence", page_icon="✦", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink:#101828; --muted:#667085; --blue:#4659e8; --violet:#7c3aed; --line:#e5e7eb; }
    .stApp { background: #f7f8fc; color: var(--ink); font-family: 'DM Sans', sans-serif; }
    .block-container { max-width: 1220px; padding: 2.2rem 2rem 4rem; }
    h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.035em; }
    [data-testid="stSidebar"] { background: #111827; border-right: 0; }
    [data-testid="stSidebar"] * { color: #f8fafc !important; }
    .brand { display:flex; align-items:center; gap:12px; margin-bottom:2.2rem; }
    .brand-mark { width:42px; height:42px; display:grid; place-items:center; border-radius:13px; background:linear-gradient(135deg,#818cf8,#c084fc); color:white; font-size:23px; font-weight:700; }
    .brand-name { font:700 21px 'Space Grotesk'; color:white; }
    .brand-sub { color:#98a2b3; font-size:11px; margin-top:2px; }
    .hero { position:relative; overflow:hidden; border-radius:28px; padding:3.2rem 3.3rem; color:white; background:radial-gradient(circle at 82% 20%,rgba(167,139,250,.7),transparent 24%), linear-gradient(120deg,#111827 0%,#2734a7 57%,#6843c5 100%); box-shadow:0 20px 45px rgba(37,40,120,.18); }
    .hero:after { content:''; position:absolute; width:260px; height:260px; right:-80px; bottom:-120px; border:1px solid rgba(255,255,255,.18); border-radius:50%; box-shadow:0 0 0 30px rgba(255,255,255,.04),0 0 0 60px rgba(255,255,255,.03); }
    .eyebrow { color:#c4b5fd; text-transform:uppercase; letter-spacing:.16em; font-size:11px; font-weight:700; margin-bottom:1rem; }
    .hero h1 { font-size:clamp(2.3rem,5vw,4.1rem); line-height:1.02; margin:0 0 1rem; color:white; }
    .hero p { max-width:590px; color:#dbe4ff; font-size:1.08rem; line-height:1.6; margin:0; }
    .hero-pill { display:inline-block; margin-top:1.6rem; padding:.45rem .75rem; border:1px solid rgba(255,255,255,.22); border-radius:999px; color:#eef2ff; font-size:.78rem; }
    .section { font:600 1.35rem 'Space Grotesk'; margin:2rem 0 .8rem; }
    .card { background:white; border:1px solid var(--line); border-radius:20px; padding:1.25rem; box-shadow:0 5px 18px rgba(16,24,40,.04); }
    .metric-card { background:white; border:1px solid var(--line); border-radius:16px; padding:1rem 1.1rem; }
    .metric-label { color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.06em; }
    .metric-value { font:700 1.65rem 'Space Grotesk'; margin-top:.3rem; }
    .stTextArea textarea { border:1px solid #d0d5dd; border-radius:14px; font-size:1rem; background:white; }
    .stButton button { border-radius:11px; font-weight:700; min-height:2.8rem; }
    .stProgress > div > div > div > div { background:linear-gradient(90deg,#4659e8,#a855f7); }
    .notice { border-radius:12px; padding:.85rem 1rem; background:#fff8e7; border:1px solid #f8d477; color:#7a4a00; font-size:.87rem; }
    .footer { text-align:center; color:#98a2b3; font-size:.8rem; margin-top:3rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark">✦</div><div><div class="brand-name">TruthLens</div><div class="brand-sub">NEWS INTELLIGENCE</div></div></div>', unsafe_allow_html=True)
    st.markdown("### Data workspace")
    st.caption("Train the screening model with the included dataset or upload your own CSV.")
    uploaded = st.file_uploader("Upload a CSV dataset", type=["csv"], help="Your CSV must contain text and label columns.")
    if uploaded:
        st.success(f"Loaded {uploaded.name}")
    st.divider()
    st.caption("TruthLens uses machine learning to identify patterns. It is not a substitute for independent fact-checking.")

st.markdown('<div class="hero"><div class="eyebrow">AI-powered credibility screening</div><h1>See the signal<br>behind the story.</h1><p>Analyze news language with a transparent model built to help you investigate claims faster and think more critically.</p><span class="hero-pill">TF-IDF · Logistic Regression · Explainable AI</span></div>', unsafe_allow_html=True)
st.markdown('<div class="notice">⚠ <b>Use responsibly:</b> A prediction is a statistical signal, not a verdict. Verify important claims using primary sources and reputable fact-checkers.</div>', unsafe_allow_html=True)

source = uploaded if uploaded else DATASET_PATH
if not uploaded and not DATASET_PATH.exists():
    st.error("No dataset found. Upload a CSV from the sidebar or add news.csv to the repository root.")
    st.code("text,label\n\"Article text here\",REAL\n\"Another article\",FAKE")
    st.stop()

@st.cache_data(show_spinner=False)
def prepare_dataset(file_bytes=None, file_name="default"):
    import io
    return load_and_prepare_data(io.BytesIO(file_bytes)) if file_bytes is not None else load_and_prepare_data(DATASET_PATH)

@st.cache_resource(show_spinner=False)
def build_model(df):
    return train_model(df)

try:
    dataset = prepare_dataset(uploaded.getvalue() if uploaded else None, uploaded.name if uploaded else "default")
    model, vectorizer, metrics = build_model(dataset)
except Exception as error:
    st.error(f"Could not train the model: {error}")
    st.info("Check that your CSV has text and label columns, with labels such as REAL/FAKE or 1/0.")
    st.stop()

st.markdown('<div class="section">Your model workspace</div>', unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
for column, label, value in [(m1,"Training examples",metrics["samples"]),(m2,"Test accuracy",f"{metrics['accuracy']:.1%}"),(m3,"REAL F1 score",f"{metrics['f1']:.1%}"),(m4,"Dataset", "Uploaded" if uploaded else "Default")]:
    column.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section">Analyze an article</div>', unsafe_allow_html=True)
text = st.text_area("Paste a headline or article", height=190, label_visibility="collapsed", placeholder="Paste a headline or article here to reveal the model's signals...")
_, action = st.columns([3, 1])
with action:
    analyze = st.button("✦ Analyze story", type="primary", use_container_width=True)

if analyze:
    if len(text.strip()) < 20:
        st.warning("Please enter at least 20 characters for a meaningful result.")
    else:
        result = predict_news(model, vectorizer, text)
        is_real = result["label"] == "REAL NEWS"
        st.markdown("<div class='section'>Analysis result</div>", unsafe_allow_html=True)
        result_col, probability_col = st.columns([1, 1])
        with result_col:
            (st.success if is_real else st.error)(f"{'✓' if is_real else '!' }  {result['label']}")
            st.metric("Model confidence", f"{result['confidence']:.1%}")
            st.progress(result["confidence"])
        with probability_col:
            st.caption("CLASS PROBABILITIES")
            st.bar_chart(result["probabilities"], height=180)

        explanation = explain_prediction(model, vectorizer, text)
        st.markdown("<div class='section'>What influenced the result?</div>", unsafe_allow_html=True)
        e1, e2 = st.columns(2)
        e1.markdown("**Signals toward REAL**")
        e1.dataframe({"Term": [x[0] for x in explanation["real"]], "Weight": [round(x[1], 4) for x in explanation["real"]]}, hide_index=True, use_container_width=True)
        e2.markdown("**Signals toward FAKE**")
        e2.dataframe({"Term": [x[0] for x in explanation["fake"]], "Weight": [round(x[1], 4) for x in explanation["fake"]]}, hide_index=True, use_container_width=True)

with st.expander("View evaluation details"):
    st.code(metrics["report"])
    st.caption("Confusion matrix — rows are actual labels, columns are predicted labels (FAKE, REAL).")
    st.dataframe(metrics["matrix"], use_container_width=True)

st.markdown('<div class="footer">TruthLens · Built for better questions, not blind certainty.</div>', unsafe_allow_html=True)

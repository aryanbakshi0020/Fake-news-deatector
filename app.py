import streamlit as st
from detector import DATASET_PATH, explain_prediction, load_and_prepare_data, predict_news, train_model

st.set_page_config(page_title="TruthLens", page_icon="📰", layout="wide")

st.markdown("<style>.block-container{max-width:1100px;padding-top:2rem}.hero{padding:2rem;border-radius:18px;background:linear-gradient(135deg,#172554,#2563eb);color:white;margin-bottom:1rem}.small{color:#64748b}</style>", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>📰 TruthLens</h1><p>AI-assisted news credibility screening with transparent predictions.</p></div>', unsafe_allow_html=True)
st.warning("This is a machine-learning screening tool, not a fact-checker. Verify important claims with trusted sources.")

if not DATASET_PATH.exists():
    st.error("news.csv is missing. Add it to the repository root before running predictions.")
    st.code("text,label\nArticle text here,REAL\nAnother article,FAKE")
    st.stop()

try:
    df = load_and_prepare_data()
    model, vectorizer, metrics = train_model(df)
except Exception as error:
    st.error(f"Could not train the model: {error}")
    st.stop()

with st.sidebar:
    st.header("Model overview")
    st.metric("Training examples", metrics["samples"])
    st.metric("Test accuracy", f"{metrics['accuracy']:.1%}")
    st.metric("REAL F1 score", f"{metrics['f1']:.1%}")
    st.caption("TF-IDF word and phrase features + balanced Logistic Regression")

st.subheader("Analyze an article")
text = st.text_area("Paste a headline or article", height=220, placeholder="Paste news text here...")
if st.button("Analyze credibility", type="primary", use_container_width=True):
    if len(text.strip()) < 20:
        st.warning("Please enter at least 20 characters for a meaningful result.")
    else:
        result = predict_news(model, vectorizer, text)
        is_real = result["label"] == "REAL NEWS"
        st.success(f"Prediction: {result['label']}") if is_real else st.error(f"Prediction: {result['label']}")
        left, right = st.columns(2)
        left.metric("Model confidence", f"{result['confidence']:.1%}")
        left.progress(result["confidence"])
        right.write("**Class probabilities**")
        right.bar_chart(result["probabilities"])
        explanation = explain_prediction(model, vectorizer, text)
        st.subheader("Words influencing this prediction")
        e1, e2 = st.columns(2)
        e1.write("**Signals toward REAL**")
        e1.dataframe({"term": [x[0] for x in explanation["real"]], "weight": [round(x[1], 4) for x in explanation["real"]]}, hide_index=True)
        e2.write("**Signals toward FAKE**")
        e2.dataframe({"term": [x[0] for x in explanation["fake"]], "weight": [round(x[1], 4) for x in explanation["fake"]]}, hide_index=True)

with st.expander("Evaluation details"):
    st.code(metrics["report"])
    st.write("Confusion matrix (rows = actual, columns = predicted):")
    st.dataframe(metrics["matrix"], use_container_width=True)

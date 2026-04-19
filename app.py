import streamlit as st
from transformers import pipeline

# Load model once
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

sentiment_pipeline = load_model()

def predict(text):
    result = sentiment_pipeline(str(text)[:512])[0]
    label = result['label'].lower()
    score = round(result['score'] * 100, 2)
    sentiment = 'negative' if 'negative' in label else 'positive'
    return sentiment, score

# UI
st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬")
st.title("💬 Tweet Sentiment Analyzer")
st.markdown("Analyze whether a tweet is **Positive** or **Negative** using a Twitter-trained RoBERTa model.")

st.markdown("---")

user_input = st.text_area("Paste a tweet or review here:", height=150,
                           placeholder="e.g. This product is absolutely amazing!")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        sentiment, confidence = predict(user_input)
        st.markdown("---")
        if sentiment == "positive":
            st.success(f"✅ Sentiment: **POSITIVE**")
        else:
            st.error(f"❌ Sentiment: **NEGATIVE**")
        st.metric("Confidence", f"{confidence}%")

st.markdown("---")
st.caption("Built by Abhishek N Salian | MSc Data Science, UE Germany")
import streamlit as st
import joblib
import pandas as pd

# Load trained model and vectorizer
model = joblib.load("fake_news_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

# --- Streamlit App ---
st.set_page_config(page_title="Fake News Detector 📰", page_icon="🕵️", layout="centered")
st.title("📰 Fake News Detector")
st.write("Paste a news article or headline below to see if it's real or fake.")

# Text input
user_input = st.text_area("Enter article text or headline", height=200)

# Predict
if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        X = vectorizer.transform([user_input])
        prediction = model.predict(X)[0]
        confidence = model.predict_proba(X)[0][prediction]

        if prediction == 1:
            st.success(f"✅ This looks **REAL** ({confidence * 100:.2f}% confident)")
        else:
            st.error(f"❌ This looks **FAKE** ({confidence * 100:.2f}% confident)")

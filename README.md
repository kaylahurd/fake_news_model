# 📰 Fake News Detector

A machine learning web app that detects whether a news article is **real** or **fake** using Natural Language Processing (NLP).


---

## 🚀 Features

- Classifies news as **REAL** or **FAKE**
- Accepts full articles or headlines as input
- Displays **prediction confidence**
- Built with:
  - Logistic Regression
  - TF-IDF vectorization
  - Streamlit for interactive web app

---

## 💡 How It Works

1. **Dataset**: Combined 44,000+ articles from Reuters (real) and known fake news websites  
2. **Preprocessing**: Cleaned text and removed stopwords  
3. **Model**: Trained a logistic regression classifier using Scikit-learn  
4. **Web App**: Built with Streamlit for live predictions  
5. **Output**: Returns whether the input is real or fake + probability

---

## 🧪 Example

```text
Input:
"BREAKING: NASA confirms Earth is flat and launches into space canceled."

Output:
❌ This looks FAKE (98.5% confident)


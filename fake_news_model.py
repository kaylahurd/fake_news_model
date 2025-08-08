# fake_news_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Load datasets
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# 2. Add labels
fake["label"] = 0  # 0 = FAKE
real["label"] = 1  # 1 = REAL

# 3. Combine and shuffle
df = pd.concat([fake, real])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 4. Use the article text as features
X = df["text"]
y = df["label"]

# 5. Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_tfidf = vectorizer.fit_transform(X)

# 6. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# 7. Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# 8. Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}\n")
print("Classification Report:\n", classification_report(y_test, y_pred))

# 9. Save model and vectorizer
joblib.dump(model, "fake_news_model.joblib")
joblib.dump(vectorizer, "tfidf_vectorizer.joblib")

print("\nModel and vectorizer saved!")

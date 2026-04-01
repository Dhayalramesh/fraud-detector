import streamlit as st
import pandas as pd
import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="UPI Fraud Detector", page_icon="🛡️")

st.title("🛡️ UPI Fraud Detector (Smart Hybrid AI)")
st.write("AI system with ML + Fraud Rules + Bank Safe Detection")

# Files
MODEL_FILE = "model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"

# Load or Train
if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)
    accuracy = None
else:
    data = pd.read_csv("data.csv")
    data = data.dropna()

    X = data["text"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)

# Show status
if accuracy is not None:
    st.success(f"📊 Model trained | Accuracy: {round(accuracy * 100, 2)}%")
else:
    st.info("⚡ Model loaded (fast mode)")

# Input
message = st.text_area("📩 Enter Message")

# 🔴 Fraud Keywords
danger_keywords = [
    "otp", "urgent", "click link", "verify account",
    "bank suspended", "update kyc",
    "send money", "upi collect", "request money"
]

# 🟢 Bank Safe Keywords
safe_keywords = [
    "debited", "credited", "balance",
    "transaction successful", "upi payment successful",
    "withdrawn", "credited to your account",
    "available balance", "avl bal"
]

def is_danger(msg):
    msg = msg.lower()
    return any(word in msg for word in danger_keywords)

def is_safe(msg):
    msg = msg.lower()
    return any(word in msg for word in safe_keywords)

# Button
if st.button("🔍 Analyze Message"):

    if message.strip() == "":
        st.warning("⚠️ Please enter a message")

    else:
        st.subheader("🔎 Result")

        # 🟢 SAFE RULE FIRST (HIGH PRIORITY)
        if is_safe(message):
            st.success("✅ Legit Bank Message (Safe)")
            st.progress(0.9)

        # 🔴 FRAUD RULE SECOND
        elif is_danger(message):
            st.error("🚨 High Risk Scam Detected (Rule-Based)")
            st.progress(0.95)

        # 🤖 ML LAST
        else:
            msg_vec = vectorizer.transform([message])
            prediction = model.predict(msg_vec)[0]
            prob = model.predict_proba(msg_vec)[0][1]

            confidence = int(prob * 100)

            if prediction == 1:
                st.error(f"🚨 Scam Detected\nConfidence: {confidence}%")
                st.progress(confidence / 100)
            else:
                st.success(f"✅ Safe Message\nConfidence: {100 - confidence}%")
                st.progress((100 - confidence) / 100)

# Footer
st.caption("AI Fraud Detection System | Hybrid ML + Rules + Banking Logic")
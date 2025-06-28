import streamlit as st
import joblib
import numpy as np

# Load model and vectorizer
model = joblib.load("models/best_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# App title
st.title("🧠 Sentiment Analysis App")
st.subheader("Enter a review below to predict its sentiment")

# Input box
user_input = st.text_area("📝 Your review", height=150, placeholder="Type your review here...")

if st.button("🔍 Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        # Transform input
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        probabilities = model.predict_proba(input_vector)[0]
        confidence = np.max(probabilities)

        # Display result
        st.markdown(f"### 🔮 Prediction: {prediction.upper()}")
        st.markdown(f"*Confidence:* {confidence*100:.2f}%")

        # Emoji feedback
        st.success("✅ Positive!") if prediction == "positive" else st.error("❌ Negative.")
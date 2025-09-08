import streamlit as st
from sentiment_analysis_production import predict_sentiment

st.title("Sentiment Analysis App")
st.markdown("Enter a product review below and click **Analyze** to see the predicted sentiment.")

review_text = st.text_area("Enter your review:")

if st.button("Analyze"):
    if review_text.strip():
        sentiment = predict_sentiment(review_text)
        st.success(f"Predicted Sentiment: {sentiment}")
    else:
        st.error("Please enter a review to analyze.")
import streamlit as st
from nlp_processor import processor

st.set_page_config(page_title="Test App", page_icon="🤖")

st.title("Simple Sentiment Test")

# Simple state
if "current_text" not in st.session_state:
    st.session_state.current_text = "I love this product!"

# Display current text
st.write("Current text:", st.session_state.current_text)

# Text area
new_text = st.text_area("Enter text:", st.session_state.current_text)

# Update button
if st.button("Update Text"):
    st.session_state.current_text = new_text
    st.rerun()

# Example buttons
st.subheader("Examples:")
if st.button("Example 1 - Positive"):
    st.session_state.current_text = "This is amazing! Love it!"
    st.rerun()

if st.button("Example 2 - Negative"):
    st.session_state.current_text = "This is terrible! Hate it!"
    st.rerun()

if st.button("Example 3 - Neutral"):
    st.session_state.current_text = "It's okay, nothing special."
    st.rerun()

# Analyze button
if st.button("Analyze Sentiment"):
    sentiment, confidence = processor.predict_sentiment(st.session_state.current_text)
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Confidence: {confidence:.0%}")

import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

# Download VADER lexicon if not already installed
nltk.download('vader_lexicon')

# Initialize VADER
sia = SentimentIntensityAnalyzer()

# Map VADER scores to emotions
def analyze_with_vader(text: str):
    scores = sia.polarity_scores(text)
    compound = scores['compound']

    if compound >= 0.5:
        return "Happy", scores
    elif compound <= -0.5:
        return "Angry", scores
    else:
        return "Sad", scores

# Streamlit UI
st.set_page_config(
    page_title="VADER Sentiment Dashboard",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Real-Time Sentiment Analyzer (VADER)")
st.markdown("*Classifies your text as Happy, Sad, or Angry using VADER*")

# Input section
st.subheader("Enter Text to Analyze")
user_input = st.text_area(
    "Type or paste your text here:",
    placeholder="Example: I'm feeling really excited about my new job!",
    height=100
)

# Buttons
col1, col2 = st.columns([1, 1])
with col1:
    analyze_button = st.button("🔍 Analyze Sentiment", type="primary")
with col2:
    clear_button = st.button("🗑️ Clear Text")

if clear_button:
    st.rerun()

if analyze_button:
    if not user_input.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing with VADER..."):
            sentiment, scores = analyze_with_vader(user_input)

            # Show main result
            if sentiment == "Happy":
                color, emoji = "#FFD700", "😊"
            elif sentiment == "Sad":
                color, emoji = "#4169E1", "😢"
            else:
                color, emoji = "#FF4500", "😠"

            st.markdown(f"""
            <div style='padding: 20px; border-radius: 10px; background-color: {color}20; border: 2px solid {color}'>
                <h2 style='color: {color}; text-align: center; margin: 0;'>
                    {emoji} Detected Sentiment: {sentiment} {emoji}
                </h2>
            </div>
            """, unsafe_allow_html=True)

            # Show VADER raw scores
            st.markdown("### 📊 VADER Score Breakdown")
            st.json(scores)

            # Convert scores to DataFrame for chart
            df = pd.DataFrame({
                "Sentiment": ["Positive", "Negative", "Neutral"],
                "Score": [scores["pos"], scores["neg"], scores["neu"]]
            })

            # Show live bar chart
            st.bar_chart(df.set_index("Sentiment"))

# Sidebar info
with st.sidebar:
    st.header("💡 How It Works")
    st.markdown("""
    - **Happy** 😊 if positivity is strong (compound ≥ 0.5)  
    - **Sad** 😢 if neutral/weak sentiment (-0.5 < compound < 0.5)  
    - **Angry** 😠 if negativity is strong (compound ≤ -0.5)  
    """)

    st.header("📊 About VADER")
    st.markdown("""
    VADER (Valence Aware Dictionary for sEntiment Reasoning) is a rule-based model
    optimized for **social media text**.  
    It provides 4 scores:
    - **Positive**
    - **Negative**
    - **Neutral**
    - **Compound** (overall sentiment, from -1 to +1)
    """)

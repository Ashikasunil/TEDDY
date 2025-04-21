
import streamlit as st
from textblob import TextBlob
import random

# ---------------- Detect Emotion ----------------
def detect_emotion(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.5:
        return "joy"
    elif polarity > 0.1:
        return "hopeful"
    elif polarity < -0.5:
        return "sadness"
    elif polarity < -0.1:
        return "anger"
    else:
        return "neutral"

# ---------------- Recommendations ----------------
daily_tips = {
    "joy": [
        "Keep a gratitude journal 📝",
        "Share your happiness with someone 🌟"
    ],
    "hopeful": [
        "Set small goals for the day 🎯",
        "Practice 10 mins of mindfulness 🌿"
    ],
    "sadness": [
        "Call a close friend or loved one 📞",
        "Listen to soothing music 🎵"
    ],
    "anger": [
        "Take a brisk walk 🚶‍♂️",
        "Try deep breathing for 5 minutes 🌬️"
    ],
    "neutral": [
        "Take a screen break ☕",
        "Organize your workspace for mental clarity 🗂️"
    ]
}

# ---------------- Regional Movie Recommendations ----------------
movie_recommendations = {
    "joy": [
        "Tamil: Oh My Kadavule (2020)",
        "Malayalam: Ustad Hotel (2012)"
    ],
    "hopeful": [
        "Tamil: Raja Rani (2013)",
        "Malayalam: Charlie (2015)"
    ],
    "sadness": [
        "Tamil: Vaaranam Aayiram (2008)",
        "Malayalam: Kumbalangi Nights (2019)"
    ],
    "anger": [
        "Tamil: 96 (2018)",
        "Malayalam: Bangalore Days (2014)"
    ],
    "neutral": [
        "Tamil: Mersal (2017)",
        "Malayalam: Premam (2015)"
    ]
}

# ---------------- Streamlit UI ----------------
def caretaker_assistant():
    st.subheader("🧑‍⚕️ Mental Wellbeing & Daily Life Assistant")

    user_note = st.text_area("How are you feeling right now?")
    
    if user_note:
        emotion = detect_emotion(user_note)
        st.markdown(f"### Detected Emotion: `{emotion.upper()}`")

        st.markdown("#### 🛠️ Personalized Daily Tips:")
        for tip in daily_tips[emotion]:
            st.markdown(f"- {tip}")

        st.markdown("#### 🎬 Feel-Good Regional Movie Suggestions:")
        for movie in movie_recommendations[emotion]:
            st.markdown(f"- {movie}")

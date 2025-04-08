
import streamlit as st
import random
from textblob import TextBlob

# ---------------- Dialogue-based Chatbot ----------------
dialogue_responses = {
    "hello": ["Hi! How are you feeling today?", "Hey there! Want to chat about your day?"],
    "how are you": ["I'm just a bot, but I'm here to listen to you!", "Doing well, thanks! Tell me about your feelings."],
    "bye": ["Goodbye! Take care of your mental health!", "See you soon! You're not alone."],
}

# ---------------- Emotion Recognition ----------------
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

emotion_activities = {
    "joy": [
        "Celebrate your good mood with this happy dance video 🎉: https://www.youtube.com/watch?v=9pUeE3eV3sM",
        "Try this feel-good playlist 😄: https://www.youtube.com/watch?v=ZbZSe6N_BXs"
    ],
    "hopeful": [
        "Keep the hope alive with this uplifting TED talk 💡: https://www.youtube.com/watch?v=mbbMLOZjUYI",
        "Breathe in calm with this meditation 🌿: https://www.youtube.com/watch?v=inpok4MKVLM"
    ],
    "sadness": [
        "Try journaling or listening to calming music 🎧: https://www.youtube.com/watch?v=3bGNuRtlI9k",
        "Watch this healing talk 🕊️: https://www.youtube.com/watch?v=XiCrniLQGYc"
    ],
    "anger": [
        "Let it out with this relaxing yoga 🔄: https://www.youtube.com/watch?v=v7AYKMP6rOE",
        "Listen to calming ambient music 🌌: https://www.youtube.com/watch?v=2OEL4P1Rz04"
    ],
    "neutral": [
        "Take a deep breath 🧘: https://www.youtube.com/watch?v=wfDTp2GogaQ",
        "How about a peaceful background mix 🎶: https://www.youtube.com/watch?v=lFcSrYw-ARY"
    ]
}

# ---------------- Streamlit UI ----------------
st.title("💬 Extended Features: Chatbot & Emotion Helper")

user_input = st.text_input("Talk to me 🙂")

if user_input:
    clean_input = user_input.lower()
    # Dialogue system
    matched = False
    for key in dialogue_responses:
        if key in clean_input:
            response = random.choice(dialogue_responses[key])
            matched = True
            break
    if not matched:
        response = "I'm here to chat, feel free to share more!"

    # Emotion Detection
    emotion = detect_emotion(user_input)
    st.markdown(f"**Chatbot:** {response}")
    st.markdown(f"🧠 Detected Emotion: `{emotion}`")

    st.subheader("🎵 Mood-Based Activities for You")
    for act in emotion_activities[emotion]:
        st.markdown(f"- {act}")

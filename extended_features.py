import streamlit as st
import random
from textblob import TextBlob

# ---------------- Dialogue-based Chatbot ----------------
dialogue_responses = {
    "hello": ["Hi! How are you feeling today?", "Hey there! Want to chat about your day?"],
    "how are you": ["I'm just a bot, but I'm here to listen to you!", "Doing well, thanks! Tell me about your feelings."],
    "bye": ["Goodbye! Take care of your mental health!", "See you soon! You're not alone."]
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
        "🎉 [Happy Dance Video](https://www.youtube.com/watch?v=9pUeE3eV3sM)",
        "😄 [Feel-Good Playlist](https://www.youtube.com/watch?v=ZbZSe6N_BXs)"
    ],
    "hopeful": [
        "💡 [Uplifting TED Talk](https://www.youtube.com/watch?v=mbbMLOZjUYI)",
        "🌿 [Meditation Session](https://www.youtube.com/watch?v=inpok4MKVLM)"
    ],
    "sadness": [
        "🎧 [Calming Music](https://www.youtube.com/watch?v=3bGNuRtlI9k)",
        "🕊️ [Healing Talk](https://www.youtube.com/watch?v=XiCrniLQGYc)"
    ],
    "anger": [
        "🔄 [Yoga to Cool Down](https://www.youtube.com/watch?v=v7AYKMP6rOE)",
        "🌌 [Ambient Relaxation Music](https://www.youtube.com/watch?v=2OEL4P1Rz04)"
    ],
    "neutral": [
        "🧘 [Breathing Exercise](https://www.youtube.com/watch?v=wfDTp2GogaQ)",
        "🎶 [Peaceful Background Mix](https://www.youtube.com/watch?v=lFcSrYw-ARY)"
    ]
}

# ---------------- Streamlit UI ----------------
def run():
    st.title("💬 Extended Features: Emotion Helper & Dialogue Bot")

    user_input = st.text_input("Talk to me 🙂")

    if user_input:
        clean_input = user_input.lower()
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
        st.markdown(f"**🤖 Chatbot:** {response}")
        st.markdown(f"🧠 **Detected Emotion:** `{emotion}`")

        st.subheader("🎵 Mood-Based Activities for You")
        for act in emotion_activities[emotion]:
            st.markdown(f"- {act}")



# Must be the first Streamlit command
import streamlit as st
st.set_page_config(page_title="Mental Health Assistant", layout="centered")

# Sidebar navigation
st.sidebar.title("🧭 Navigation")
app_mode = st.sidebar.radio("Choose a section", [
    "Home",
    "Chatbot + Sentiment",
    "Mood Timeline",
    "Chat History",
    "Survey Analysis",
    "🎯 Extended Features",
    "🧠 Caretaker Assistant"
])

# Imports
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Import modules
try:
    import extended_features
except:
    extended_features = None

try:
    from caretaker_module import caretaker_assistant
except:
    caretaker_assistant = None

# --- Intent Model Setup ---
intent_data = {
    "text": [
        "I feel stressed", "I'm anxious", "Hello", "Hi",
        "I'm sad", "Feeling low", "I'm overwhelmed", "Good morning"
    ],
    "intent": [
        "stress", "anxiety", "greeting", "greeting",
        "sadness", "sadness", "stress", "greeting"
    ]
}
df_intent = pd.DataFrame(intent_data)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df_intent['text'])
clf = LogisticRegression()
clf.fit(X, df_intent['intent'])

# Responses & Recommendations
responses = {
    "greeting": ["Hi there! How can I support you today?"],
    "stress": ["Stress is tough. Let’s take a deep breath together."],
    "anxiety": ["It’s okay to feel anxious. I'm here with you."],
    "sadness": ["I’m sorry you feel this way. Want to talk more about it?"],
    "default": ["Tell me more, I'm listening."]
}
recommendations = {
    "stress": [
        {"title": "5-Minute Meditation", "url": "https://www.youtube.com/watch?v=inpok4MKVLM"},
        {"title": "Stretch & Relax", "url": "https://www.youtube.com/watch?v=qHJ992N-Dhs"}
    ],
    "anxiety": [
        {"title": "Anxiety Relief Music", "url": "https://www.youtube.com/watch?v=ZToicYcHIOU"},
        {"title": "Grounding Exercise", "url": "https://www.youtube.com/watch?v=KZXT7L4s0bY"}
    ],
    "sadness": [
        {"title": "Uplifting Music", "url": "https://www.youtube.com/watch?v=UfcAVejslrU"},
        {"title": "Talk on Depression", "url": "https://www.youtube.com/watch?v=XiCrniLQGYc"}
    ],
    "default": [
        {"title": "Mental Health Playlist", "url": "https://www.youtube.com/playlist?list=PLFzWFredxyJlR9L1_LPODw_JH6XkUnYVX"}
    ]
}
crisis_keywords = ["suicide", "kill myself", "end it", "hopeless", "give up"]
log_data = []

def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "POSITIVE", polarity
    elif polarity < 0:
        return "NEGATIVE", polarity
    else:
        return "NEUTRAL", polarity

def process_input(text):
    label, score = get_sentiment(text)
    if any(word in text.lower() for word in crisis_keywords):
        return "[⚠️ Crisis Detected] Please seek immediate help.", "Crisis"
    input_vec = vectorizer.transform([text])
    intent = clf.predict(input_vec)[0] if clf.predict_proba(input_vec).max() > 0.4 else "default"
    response = random.choice(responses[intent])
    log_data.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "user_input": text,
        "sentiment": label,
        "sentiment_score": score,
        "intent": intent,
        "response": response
    })
    return f"[{label}] {response}", intent

# --- Streamlit Pages ---
if app_mode == "Home":
    st.title("🏠 Welcome to the Mental Health Assistant")
    st.write("Select a feature from the sidebar to begin your personalized mental wellness journey.")

elif app_mode == "Chatbot + Sentiment":
    st.title("💬 Chatbot and Sentiment Analysis")
    user_input = st.text_input("How are you feeling today?")
    if user_input:
        output, intent = process_input(user_input)
        st.markdown(f"**Bot:** {output}")
        st.subheader("🎯 Personalized Recommendations")
        for rec in recommendations.get(intent, recommendations["default"]):
            st.markdown(f"- [{rec['title']}]({rec['url']})")

elif app_mode == "Mood Timeline":
    st.title("📈 Mood Timeline Visualization")
    if log_data:
        df_log = pd.DataFrame(log_data)
        df_log["sentiment_num"] = df_log["sentiment"].map({
            "POSITIVE": 1,
            "NEGATIVE": -1,
            "NEUTRAL": 0
        })
        st.line_chart(df_log.set_index("time")["sentiment_num"])
    else:
        st.info("Start chatting to see your mood timeline.")

elif app_mode == "Chat History":
    st.title("📜 Chat History Log")
    if log_data:
        st.dataframe(pd.DataFrame(log_data))
    else:
        st.info("No conversation history yet.")

elif app_mode == "Survey Analysis":
    st.title("📊 Survey Data Analysis")
    survey_file = st.file_uploader("Upload your Google Form survey CSV", type="csv")
    if survey_file:
        survey_df = pd.read_csv(survey_file)
        st.write("📝 Survey Preview", survey_df.head())
        if "Rate your overall mood (1–5)" in survey_df.columns:
            avg_mood = survey_df["Rate your overall mood (1–5)"].mean()
            st.metric("🌤️ Average Mood Score", round(avg_mood, 2))
            st.subheader("Mood Score Distribution")
            fig, ax = plt.subplots()
            sns.histplot(survey_df["Rate your overall mood (1–5)"], bins=5, ax=ax)
            st.pyplot(fig)
        if "What are you struggling with lately?" in survey_df.columns:
            text = " ".join(survey_df["What are you struggling with lately?"].dropna())
            wc = WordCloud(width=800, height=400, background_color='white').generate(text)
            st.subheader("🧠 Common Concerns")
            st.image(wc.to_array())

elif app_mode == "🎯 Extended Features":
    st.title("✨ Extended Features")
    if extended_features:
        extended_features.show_extended_features()
    else:
        st.warning("Extended features module not found.")

elif app_mode == "🧠 Caretaker Assistant":
    st.title("🧠 Smart Caretaker Assistant")
    if caretaker_assistant:
        caretaker_assistant()
    else:
        st.warning("Caretaker assistant module not found.")







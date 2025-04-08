# Imports
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import extended_features  # if you're using the extra module
# Any other libraries...

# ✅ This must be the VERY FIRST Streamlit command
st.set_page_config(page_title="Mental Health Assistant", layout="centered")

# All other Streamlit UI logic starts AFTER this
st.sidebar.title("🧭 Navigation")
app_mode = st.sidebar.radio("Choose a section", [
    "Home",
    "Chatbot + Sentiment",
    "Mood Timeline",
    "Chat History",
    "Survey Analysis",
    "🎯 Extended Features"
])

# Your conditional logic to show content
if app_mode == "Home":
    st.title("🏠 Welcome to the Mental Health Assistant")
    st.write("Select a feature from the sidebar to begin.")

elif app_mode == "🎯 Extended Features":
    import extended_features

# app.py

# Standard imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import extended_features  # If you have this module

# ✅ IMPORTANT: Must be the FIRST Streamlit command
st.set_page_config(
    page_title="Mental Health Assistant",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Now continue with your Streamlit layout
st.sidebar.title("🧭 Navigation")
app_mode = st.sidebar.radio("Choose a section", [
    "Home",
    "Chatbot + Sentiment",
    "Mood Timeline",
    "Chat History",
    "Survey Analysis",
    "🎯 Extended Features"
])

import streamlit as st  # ✅ This must be first
st.set_page_config(page_title="Mental Health Assistant", layout="centered")  # ✅ This must be second

# Other imports come after
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import extended_features  # Optional, only if you have this module

st.sidebar.title("🧭 Navigation")
app_mode = st.sidebar.radio("Choose a section", [...])
# ✅ Nothing before this
import streamlit as st

# ✅ Page config must be right after the first import
st.set_page_config(page_title="Mental Health Assistant", layout="centered")

# ✅ Then continue with other imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import extended_features  # Only if this does NOT call st.* immediately

# ✅ Streamlit UI begins here
st.sidebar.title("🧭 Navigation")
app_mode = st.sidebar.radio("Choose a section", [
    "Home",
    "Chatbot + Sentiment",
    "Mood Timeline",
    "Chat History",
    "Survey Analysis",
    "🎯 Extended Features"
])
def show_extended_features():
    st.write("Here's something cool!")
# ✅ Nothing should come before this
import streamlit as st

# ✅ Must be the FIRST Streamlit command
st.set_page_config(
    page_title="Mental Health Assistant",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ✅ Now safe to import other libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import base64
from wordcloud import WordCloud
from transformers import pipeline

# ✅ Safe to import custom modules if they don’t use Streamlit at the top-level
try:
    import extended_features  # Make sure this does not run st.* directly on import
except:
    st.warning("Extended features module not loaded")

# ✅ Streamlit UI starts here
st.sidebar.title("🧭 Navigation")
app_mode = st.sidebar.radio("Choose a section", [
    "Home",
    "Chatbot + Sentiment",
    "Mood Timeline",
    "Chat History",
    "Survey Analysis",
    "🎯 Extended Features"
])
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
# (other imports...)

# ✅ This must be FIRST
st.set_page_config(page_title="Mental Health Assistant", layout="centered")

# Then your Streamlit UI setup
st.sidebar.title("🧭 Navigation")
app_mode = st.sidebar.radio("Choose a section", [...])
import streamlit as st

# Sidebar Navigation
st.sidebar.title("🧭 Navigation")
app_mode = st.sidebar.radio("Choose a section", [
    "Home",
    "Chatbot + Sentiment",
    "Mood Timeline",
    "Chat History",
    "Survey Analysis",
    "🎯 Extended Features"
])

# Sections
if app_mode == "Home":
    st.title("🏠 Welcome to the Mental Health Assistant")
    st.write("Select a feature from the sidebar to begin.")

elif app_mode == "Chatbot + Sentiment":
    st.title("💬 Chatbot and Sentiment Analysis")
    # Add your chatbot code here (or import a file)

elif app_mode == "Mood Timeline":
    st.title("📈 Mood Timeline Visualization")
    # Add timeline code or placeholder
    st.info("Mood timeline feature under construction.")

elif app_mode == "Chat History":
    st.title("📜 Chat History Log")
    # Add chat history code or placeholder
    st.info("Chat history feature coming soon.")

elif app_mode == "Survey Analysis":
    st.title("📊 Survey Results")
    # Add your Google Form data analysis code
    st.info("Survey analysis not yet implemented.")

elif app_mode == "🎯 Extended Features":
    import extended_features  # This imports your added module

import streamlit as st
from textblob import TextBlob
import random
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

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
df = pd.DataFrame(intent_data)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
clf = LogisticRegression()
clf.fit(X, df['intent'])

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

# Function to process input
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

# --- Streamlit Interface ---
st.set_page_config(page_title="Mental Health Assistant", layout="centered")
st.title("🧠 Personalized Mental Health Assistant")

user_input = st.text_input("How are you feeling today?")
if user_input:
    output, intent = process_input(user_input)
    st.markdown(f"**Bot:** {output}")

    st.subheader("🎯 Personalized Recommendations")
    for rec in recommendations.get(intent, recommendations["default"]):
        st.markdown(f"- [{rec['title']}]({rec['url']})")

# Chat History
if st.checkbox("📜 Show Chat History"):
    if log_data:
        st.dataframe(pd.DataFrame(log_data))
    else:
        st.info("No conversation history yet.")

# Mood Timeline
if st.checkbox("📈 Show Mood Timeline"):
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

# Survey Data Analysis
st.header("📊 Survey Data Analysis")
survey_file = st.file_uploader("Upload your Google Form survey CSV", type="csv")

if survey_file is not None:
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

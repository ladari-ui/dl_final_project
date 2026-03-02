import streamlit as st
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import tempfile
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pydub import AudioSegment

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Mental Health Emotion Analyzer",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("my_finetuned_model")
    model = AutoModelForSequenceClassification.from_pretrained("my_finetuned_model")
    return tokenizer, model

tokenizer, model = load_model()

# emotion labels
label_map = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

emoji_map = {
    "joy": "😄",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😲"
}

tips = {
    "sadness": "Try talking to a friend or taking a short walk 🌿",
    "anger": "Take deep breaths and relax 🧘",
    "fear": "Focus on slow breathing 💙",
    "joy": "Spread your happiness 🌞",
    "love": "Share your feelings with someone you trust ❤️",
    "surprise": "Take a moment to process the unexpected 😊"
}

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "🧠 Emotion Analyzer", "📈 Prediction", "📊 About", "💬 Contact"]
)

# =====================================================
# HOME
# =====================================================
if page == "🏠 Home":
    st.title("💙 Mental Health Emotion Analyzer")
    st.write("Analyze emotions from text or speech using AI.")
    st.info("Use the sidebar to start analyzing your emotions.")

# =====================================================
# EMOTION ANALYZER (VOICE + TEXT)
# =====================================================
elif page == "🧠 Emotion Analyzer":

    st.title("🧠 Emotion Analyzer")
    st.write("Speak or type your feelings below 👇")

    user_input = ""

    # 🎤 RECORD AUDIO
    audio = mic_recorder(
        start_prompt="🎤 Start Recording",
        stop_prompt="Stop Recording",
        key="recorder",
    )

    if audio is not None:
        st.info("Processing speech...")

        webm_path = None
        wav_path = None

        try:
            # save webm
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
                tmp.write(audio["bytes"])
                webm_path = tmp.name

            # convert to wav
            sound = AudioSegment.from_file(webm_path)
            wav_path = webm_path.replace(".webm", ".wav")
            sound.export(wav_path, format="wav")

            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)

            user_input = recognizer.recognize_google(audio_data, language="en-IN")
            st.success("You said: " + user_input)

        except sr.UnknownValueError:
            st.error("❌ Could not understand audio. Speak clearly.")
        except sr.RequestError:
            st.error("❌ Internet issue or Google API unavailable.")
        except Exception as e:
            st.error(f"Audio error: {e}")

        finally:
            if webm_path and os.path.exists(webm_path):
                os.remove(webm_path)
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)

    # text input
    text_input = st.text_area("Or type your text here:")
    if text_input:
        user_input = text_input

    if st.button("Analyze Emotion"):

        if user_input.strip() == "":
            st.warning("Please speak or type something.")
        else:
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)

            probs = torch.nn.functional.softmax(outputs.logits, dim=1).detach().numpy()[0]
            pred_class = np.argmax(probs)
            prediction = label_map[pred_class]

            st.success(f"Detected Emotion: {prediction} {emoji_map[prediction]}")

            if prediction == "joy":
                st.balloons()

            st.info(tips[prediction])

# =====================================================
# 📈 PREDICTION PAGE (CONFIDENCE GRAPH)
# =====================================================
elif page == "📈 Prediction":

    st.title("📈 Emotion Prediction & Confidence")
    st.write("Enter text to view prediction confidence levels")

    text = st.text_area("Type your text")

    if st.button("Predict Emotion"):

        if text.strip() == "":
            st.warning("Please enter text")
        else:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)

            probs = torch.nn.functional.softmax(outputs.logits, dim=1).detach().numpy()[0]
            pred_class = np.argmax(probs)
            emotion = label_map[pred_class]
            confidence = probs[pred_class] * 100

            st.success(f"Prediction: {emotion.upper()} {emoji_map[emotion]}")
            st.info(f"Confidence: {confidence:.2f}%")

            st.subheader("Emotion Confidence Graph")

            fig = plt.figure()
            plt.bar(label_map.values(), probs)
            plt.xlabel("Emotions")
            plt.ylabel("Confidence")
            plt.xticks(rotation=45)
            st.pyplot(fig)

# =====================================================
# ABOUT
# =====================================================
elif page == "📊 About":
    st.title("📊 About This Project")
    st.write("""
    This project uses:
    - Streamlit
    - Hugging Face Transformers
    - Deep Learning NLP
    - Speech Recognition

    It analyzes text or speech and predicts emotional state.
    """)
    st.success("Built for Mental Health Awareness 💙")

# =====================================================
# CONTACT
# =====================================================
elif page == "💬 Contact":
    st.title("💬 Contact & Feedback")

    name = st.text_input("Your Name")
    feedback = st.text_area("Your Feedback")

    if st.button("Submit"):
        if name and feedback:
            st.success("Thank you for your feedback! 💙")
        else:
            st.warning("Please fill all fields.")
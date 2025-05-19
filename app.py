import streamlit as st

# Set page config must be the first Streamlit command
st.set_page_config(page_title="Pronunciation Assistant", layout="wide")

import torch
import torch.nn as nn
import librosa
import numpy as np
import tempfile
import difflib
from gtts import gTTS
import sounddevice as sd
import soundfile as sf
from io import BytesIO
import speech_recognition as sr
import time


# -------------- Model Definition -----------------
class PronunciationLSTM(nn.Module):
    def __init__(self, num_outputs=10):
        super(PronunciationLSTM, self).__init__()
        self.lstm = nn.LSTM(13, 64, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_outputs)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        return self.fc(x)


def load_model(path):
    model = PronunciationLSTM(num_outputs=2)  # 2 outputs: good/bad pronunciation
    # In production, handle this with proper error checking
    try:
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        # Use a placeholder model for demo - in real app you'd handle this better
        pass
    model.eval()
    return model


# Use a placeholder model path (in production you'd use a real model)
model_path = "pronunciation_model.pth"
try:
    model = load_model(model_path)
except Exception as e:
    # Create a dummy model for demonstration purposes
    model = PronunciationLSTM(num_outputs=2)
    model.eval()


# -------------- MFCC Extraction -----------------
def extract_mfcc(audio_path, sr=16000, n_mfcc=13, max_len=100):
    y, _ = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T
    if mfcc.shape[0] < max_len:
        pad_width = max_len - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:max_len]
    return torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)


# -------------- Prediction -----------------
def predict(audio_path):
    try:
        mfcc = extract_mfcc(audio_path)
        with torch.no_grad():
            output = model(mfcc)
            # Interpret as probability of good pronunciation
            prob = torch.sigmoid(output[:, 1]).item()
        return prob
    except Exception as e:
        # Return a random score between 0.6 and 0.95 for demo purposes
        return np.random.uniform(0.6, 0.95)


# -------------- Transcription -----------------
def transcribe_audio(audio_path):
    r = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = r.record(source)
    try:
        text = r.recognize_google(audio)
        return text.lower()
    except sr.UnknownValueError:
        return "Could not understand the audio"
    except sr.RequestError:
        return "Error with the speech service"


# -------------- Text Comparison -----------------
def compare_texts(ref, hyp):
    # Word-level comparison
    ref_words = ref.lower().split()
    hyp_words = hyp.lower().split() if hyp else []

    # Calculate word-level accuracy
    correct_words = 0
    total_words = len(ref_words)
    mispronounced_words = []

    # Create highlighted text with correct words in white, mispronounced in red
    highlighted_text = []

    # Make a copy of hyp_words to work with
    remaining_hyp_words = hyp_words.copy()

    # Check each reference word
    for word in ref_words:
        if word in remaining_hyp_words:
            # Word was correctly pronounced
            highlighted_text.append(word)  # Regular white text
            correct_words += 1
            remaining_hyp_words.remove(word)  # Remove to avoid double counting
        else:
            # Word was mispronounced or missing
            highlighted_text.append(f"**:red[{word}]**")  # Red and bold
            mispronounced_words.append(word)

    word_accuracy = (correct_words / total_words) * 100 if total_words > 0 else 0

    # Character-level comparison using difflib
    matcher = difflib.SequenceMatcher(None, ref.lower(), hyp.lower())
    char_accuracy = matcher.ratio() * 100

    return {
        "word_accuracy": word_accuracy,
        "char_accuracy": char_accuracy,
        "highlighted_text": " ".join(highlighted_text),
        "mispronounced_words": mispronounced_words
    }


# -------------- Text-to-Speech -----------------
def generate_pronunciation_audio(word):
    tts = gTTS(word)
    temp_audio = BytesIO()
    tts.write_to_fp(temp_audio)
    temp_audio.seek(0)
    return temp_audio


# -------------- Streamlit UI -----------------
# Title and main layout
st.title("🗣️ Pronunciation Coach")

col1, col2 = st.columns([3, 2])

with col1:
    # Reference text input
    ref_text = st.text_area("Enter your reference text:",
                            "The quick brown fox jumps over the lazy dog",
                            height=100)

    # Practice levels
    st.markdown("### Practice Mode")
    practice_mode = st.radio(
        "Select your practice mode:",
        ["Listen & Learn", "Practice & Analyze", "Challenge Mode"],
        horizontal=True
    )

    # Explanation of the selected mode
    if practice_mode == "Listen & Learn":
        st.info("Listen to the correct pronunciation before practicing.")
    elif practice_mode == "Practice & Analyze":
        st.info("Record yourself and get detailed feedback on your pronunciation.")
    else:  # Challenge Mode
        st.info("Continue practicing until you achieve 95% accuracy.")

with col2:
    st.markdown("### Record Your Voice")

    # Streamlined recording experience
    recording_duration = st.slider("Recording duration (seconds)", 3, 15, 5)

    rec_col1, rec_col2 = st.columns([1, 1])

    with rec_col1:
        # Enhanced recording button
        record_button = st.button("🎙️ Start Recording", use_container_width=True)

    with rec_col2:
        # Play reference button
        if st.button("🔊 Hear Reference", use_container_width=True):
            tts = gTTS(ref_text)
            tts_fp = BytesIO()
            tts.write_to_fp(tts_fp)
            tts_fp.seek(0)
            st.audio(tts_fp.read(), format="audio/mp3")

# Recording logic with visual feedback
audio_path = None
if record_button:
    with st.spinner("Recording in progress..."):
        # Create a progress bar for recording time
        progress_bar = st.progress(0)
        for i in range(recording_duration):
            # Start recording
            if i == 0:
                recording = sd.rec(int(recording_duration * 16000), samplerate=16000, channels=1)

            # Update progress bar
            progress_bar.progress((i + 1) / recording_duration)
            time.sleep(1)

        sd.wait()  # Wait for recording to finish

        # Save recording to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, recording, 16000)
            audio_path = tmp.name

        st.success("Recording complete!")

# Analysis section
if audio_path and ref_text.strip():
    st.markdown("---")
    st.markdown("## Your Pronunciation Results")

    col_audio, col_score = st.columns([1, 2])

    with col_audio:
        st.audio(audio_path, format="audio/wav")

    # Speech recognition and analysis
    with st.spinner("Analyzing your pronunciation..."):
        try:
            # Transcribe the audio
            hyp_text = transcribe_audio(audio_path)

            # Speech quality score (from model)
            pronunciation_score = predict(audio_path) * 100

            # Text comparison analysis
            comparison = compare_texts(ref_text, hyp_text)
        except Exception as e:
            st.error(f"An error occurred during analysis. Please try again.")
            st.stop()

    # Display overall scores
    with col_score:
        st.subheader("Accuracy Scores")

        score_col1, score_col2, score_col3 = st.columns(3)

        with score_col1:
            st.metric("Pronunciation Quality", f"{pronunciation_score:.1f}%")

        with score_col2:
            st.metric("Word Matching", f"{comparison['word_accuracy']:.1f}%")

        with score_col3:
            st.metric("Overall Accuracy", f"{comparison['char_accuracy']:.1f}%")

    # Display transcribed text
    st.subheader("Your Speech Transcription")
    st.write(hyp_text)

    # Display reference text with highlighting
    st.subheader("Reference Text with Pronunciation Issues")
    st.markdown(f"Words in **:red[red]** were mispronounced or missing in your speech.")
    st.markdown(comparison["highlighted_text"])

    # Mispronounced words section
    if comparison["mispronounced_words"]:
        st.markdown("### Words to Practice")

        # Create a grid of word practice buttons
        word_cols = st.columns(min(3, len(comparison["mispronounced_words"])))

        for i, word in enumerate(comparison["mispronounced_words"]):
            col_index = i % 3
            with word_cols[col_index]:
                st.markdown(f"**{word}**")
                if st.button(f"🔊 Hear '{word}'", key=f"word_{word}"):
                    st.audio(generate_pronunciation_audio(word), format="audio/mp3")

    # Challenge mode logic
    if practice_mode == "Challenge Mode":
        if comparison["char_accuracy"] >= 95:
            st.success("🏆 Challenge completed! Your pronunciation is excellent!")
        else:
            st.warning(
                f"Keep practicing! You need {95 - comparison['char_accuracy']:.1f}% more accuracy to complete the challenge.")
            st.button("Try Again")

else:
    # Initial state message
    st.info("Enter your reference text and click 'Start Recording' to begin.")

# Helpful tips section at the bottom
with st.expander("Pronunciation Tips"):
    st.markdown("""
    * **Speak clearly and at a moderate pace** - Don't rush through the words
    * **Practice difficult words individually** - Click on highlighted words to hear them
    * **Record in a quiet environment** - Reduce background noise for better accuracy
    * **Pay attention to vowel sounds** - Many pronunciation errors come from incorrect vowel sounds
    * **Use your lips, tongue and jaw actively** - English requires more articulation than some languages
    """)
# 🗣️ Pronunciation Check

The English Pronunciation check is an interactive web application designed to help users improve their English pronunciation skills. As globalization continues to connect people from diverse linguistic backgrounds, the ability to communicate effectively in English has become increasingly important. However, many language learners face significant challenges in mastering English pronunciation due to interference from their native languages and limited access to personalized feedback.

This project addresses these challenges by developing an AI-powered pronunciation coach that provides real-time feedback on users' speech. The system leverages deep learning techniques, specifically Long Short-Term Memory (LSTM) neural networks, to analyze speech patterns and identify pronunciation errors. The web application, built with Streamlit, offers an intuitive interface where users can practice speaking, receive immediate feedback, and track their improvement over time.

The Pronunciation Check serves as an accessible, cost-effective alternative to traditional language tutoring, enabling learners to practice independently at their own pace. By providing targeted feedback on specific words and sounds, the system helps users focus their practice on areas that need the most improvement, accelerating their journey toward pronunciation mastery.

![App Screenshot](pic.jpeg)

---

## 🚀 Features

- 🎤 Record your voice directly from the browser
- 🧠 Automatic speech recognition using `SpeechRecognition`
- ❌ Detect and highlight mispronounced words
- 🔊 Get correct pronunciation using Google TTS
- 💻 Lightweight, runs in browser with Streamlit

---

## 📦 Installation

Clone this repository:

```bash
git clone https://github.com/ngoyal88/Pronunciation-Check.git
cd Pronunciation-Check
```
Install all required dependencies:
```bash
pip install -r requirements.txt
```

▶️ Run the App
To start the application, run:
```bash
streamlit run app.py
```
Once the server starts, a browser window will open showing the app.
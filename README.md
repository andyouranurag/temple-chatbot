# Temple Chatbot 🛕✨

A Streamlit-based chatbot that answers questions about Indian temples.
Features:
- Semantic search using SentenceTransformers
- Multilingual support (deep-translator + MarianMT fallback)
- Voice input (SpeechRecognition + PyAudio)
- Text-to-Speech (gTTS / Cloud TTS)
- MongoDB backend for temple data
- In-browser microphone (streamlit-webrtc)
- Map & image display

## 🚀 Run Locally

```bash
git clone https://github.com/your-username/temple-chatbot.git
cd temple-chatbot
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
streamlit run streamlit_temple.py

import streamlit as st
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from gtts import gTTS
import os
import tempfile
from deep_translator import GoogleTranslator
from streamlit_mic_recorder import mic_recorder
import pymongo
import requests

# ------------------ MongoDB Connection ------------------
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
client = pymongo.MongoClient(MONGO_URI)
db = client["templeDB"]
temple_collection = db["temples"]

# ------------------ Embedding Model ------------------
# Force CPU to avoid CUDA-related errors
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# ------------------ Utility Functions ------------------
def load_dataset():
    data = list(temple_collection.find({}, {"_id": 0}))
    if not data:  # fallback if DB empty
        sample_data = [
            {
                "name": "Kashi Vishwanath Temple",
                "location": "Varanasi, Uttar Pradesh",
                "description": "One of the twelve Jyotirlingas dedicated to Lord Shiva.",
                "history": "Rebuilt several times, last by Ahilyabai Holkar in 1780.",
                "architecture": "Nagara style with a golden spire.",
                "unique_features": "Golden domes, near Ganga river.",
                "latitude": 25.3109,
                "longitude": 83.0095,
                "image_url": "https://upload.wikimedia.org/wikipedia/commons/0/0a/Kashi_Vishwanath_Temple.jpg"
            }
        ]
        temple_collection.insert_many(sample_data)
        data = sample_data
    return data

@st.cache_resource
def build_index(docs):
    embeddings = model.encode(docs)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, embeddings

def prepare_docs(data):
    docs = []
    for t in data:
        docs.append(f"{t['name']} - {t['location']}. {t['description']} History: {t['history']} Architecture: {t['architecture']} Unique: {t['unique_features']}")
    return docs

def search(query, index, docs, data):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), k=1)
    result = data[I[0][0]]
    return result

# ------------------ Translation ------------------
def translate_text(text, target_lang="en"):
    try:
        return GoogleTranslator(source="auto", target=target_lang).translate(text)
    except Exception as e:
        return text

# ------------------ TTS ------------------
def text_to_speech(text, lang="en"):
    tts = gTTS(text=text, lang=lang)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        return fp.name

# ------------------ Streamlit UI ------------------
st.title("🛕 Temple Chatbot")

st.sidebar.header("Settings")
language = st.sidebar.selectbox("Choose Response Language", ["en", "hi", "mr", "gu", "ta", "te"])

# Load dataset from MongoDB
data = load_dataset()
docs = prepare_docs(data)
index, embeddings = build_index(docs)

# ------------------ Text Input ------------------
user_input = st.text_input("Ask me about a temple:")
if user_input:
    translated_query = translate_text(user_input, "en")
    result = search(translated_query, index, docs, data)

    # Translate back to selected language
    result_display = {
        "Name": result.get("name", ""),
        "Location": result.get("location", ""),
        "Description": translate_text(result.get("description", ""), language),
        "History": translate_text(result.get("history", ""), language),
        "Architecture": translate_text(result.get("architecture", ""), language),
        "Unique Features": translate_text(result.get("unique_features", ""), language)
    }

    st.subheader(result_display["Name"])
    st.write(f"📍 Location: {result_display['Location']}")
    st.write(f"🏛️ Architecture: {result_display['Architecture']}")
    st.write(f"📖 History: {result_display['History']}")
    st.write(f"✨ Unique Features: {result_display['Unique Features']}")

    if result.get("image_url"):
        st.image(result["image_url"], caption=result_display["Name"], use_container_width=True)

    if result.get("latitude") and result.get("longitude"):
        maps_api = os.getenv("GOOGLE_MAPS_API_KEY")
        if maps_api:
            map_url = f"https://maps.googleapis.com/maps/api/staticmap?center={result['latitude']},{result['longitude']}&zoom=14&size=600x400&markers=color:red%7C{result['latitude']},{result['longitude']}&key={maps_api}"
            st.image(map_url, caption="Temple Location Map")

    # TTS
    tts_file = text_to_speech(" ".join(result_display.values()), lang=language)
    audio_file = open(tts_file, "rb")
    st.audio(audio_file.read(), format="audio/mp3")

# ------------------ Voice Input ------------------
st.subheader("🎤 Voice Input")

voice_text = mic_recorder(
    start_prompt="🎙️ Start Recording",
    stop_prompt="⏹️ Stop Recording",
    just_once=True,
    use_container_width=True
)

if voice_text:
    st.session_state["voice_input"] = voice_text
    st.success(f"Recognized Voice Input: {voice_text}")

    translated_query = translate_text(voice_text, "en")
    result = search(translated_query, index, docs, data)

    # Translate back to selected language
    result_display = {
        "Name": result.get("name", ""),
        "Location": result.get("location", ""),
        "Description": translate_text(result.get("description", ""), language),
        "History": translate_text(result.get("history", ""), language),
        "Architecture": translate_text(result.get("architecture", ""), language),
        "Unique Features": translate_text(result.get("unique_features", ""), language)
    }

    st.subheader(result_display["Name"])
    st.write(f"📍 Location: {result_display['Location']}")
    st.write(f"🏛️ Architecture: {result_display['Architecture']}")
    st.write(f"📖 History: {result_display['History']}")
    st.write(f"✨ Unique Features: {result_display['Unique Features']}")

    if result.get("image_url"):
        st.image(result["image_url"], caption=result_display["Name"], use_container_width=True)

    if result.get("latitude") and result.get("longitude"):
        maps_api = os.getenv("GOOGLE_MAPS_API_KEY")
        if maps_api:
            map_url = f"https://maps.googleapis.com/maps/api/staticmap?center={result['latitude']},{result['longitude']}&zoom=14&size=600x400&markers=color:red%7C{result['latitude']},{result['longitude']}&key={maps_api}"
            st.image(map_url, caption="Temple Location Map")

    # TTS
    tts_file = text_to_speech(" ".join(result_display.values()), lang=language)
    audio_file = open(tts_file, "rb")
    st.audio(audio_file.read(), format="audio/mp3")

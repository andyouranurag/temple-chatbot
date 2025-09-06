import streamlit as st
import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np
try:
    import faiss
except Exception as e:
    faiss = None

from deep_translator import GoogleTranslator
from gtts import gTTS
import tempfile
import base64
import speech_recognition as sr
from pydub import AudioSegment
from pymongo import MongoClient

st.set_page_config(page_title="Temple Chatbot", layout="wide")

# ----------------------------- MongoDB Atlas -----------------------------
@st.cache_resource
def get_mongo_collection():
    client = MongoClient(st.secrets["MONGO_URI"])
    db = client.get_database()
    return db["temples"]

temples_collection = get_mongo_collection()

# ----------------------------- Helpers -----------------------------
@st.cache_resource
def load_embedding_model(name="all-MiniLM-L6-v2"):
    return SentenceTransformer(name)

@st.cache_data
def build_documents(temples):
    docs = []
    for t in temples:
        docs.append({
            "name": t.get("name", ""),
            "location": t.get("location", ""),
            "description": t.get("description", ""),
            "history": t.get("history", ""),
            "architecture": t.get("architecture", ""),
            "unique_features": t.get("unique_features", ""),
            "raw": f"{t.get('name','')} - {t.get('location','')}. {t.get('description','')} "
                   f"History: {t.get('history','')} Architecture: {t.get('architecture','')} "
                   f"Unique: {t.get('unique_features','')}"
        })
    return docs

@st.cache_resource
def build_faiss_index(embeddings):
    d = embeddings.shape[1]
    if faiss is None:
        raise RuntimeError("faiss is not available. Please install faiss-cpu.")
    idx = faiss.IndexFlatL2(d)
    idx.add(np.array(embeddings).astype('float32'))
    return idx

def tts_and_audio_bytes(text, lang="en"):
    tts = gTTS(text=text, lang=lang)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    with open(tmp.name, "rb") as f:
        b = f.read()
    try:
        os.unlink(tmp.name)
    except:
        pass
    return b

def transcribe_audio_file(uploaded_file):
    r = sr.Recognizer()
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_in:
        tmp_in.write(uploaded_file.getvalue())
        tmp_in.flush()
        fname = tmp_in.name

    wav_path = fname
    if ext != ".wav":
        try:
            sound = AudioSegment.from_file(fname)
            wav_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            sound.export(wav_tmp.name, format="wav")
            wav_path = wav_tmp.name
        except Exception as e:
            return None, f"Error converting audio: {e}"

    with sr.AudioFile(wav_path) as source:
        audio = r.record(source)
    try:
        text = r.recognize_google(audio)
    except Exception as e:
        return None, f"Transcription error: {e}"
    finally:
        try:
            os.unlink(fname)
        except: pass
        if wav_path != fname:
            try: os.unlink(wav_path)
            except: pass
    return text, None

def translate_text(text, target_lang="en"):
    try:
        return GoogleTranslator(source="auto", target=target_lang).translate(text)
    except Exception as e:
        st.warning(f"Translation failed: {e}")
        return text

# ----------------------------- UI Layout -----------------------------
st.title("Temple Dataset Chatbot — Streamlit")
st.markdown("A lightweight chatbot that searches your temple dataset using semantic embeddings, supports translation, structured responses, and basic voice I/O.")

col1, col2 = st.columns([2,1])

with col2:
    st.header("Settings")
    uploaded = st.file_uploader("Upload temples.json (optional)", type=['json'])
    use_sample = st.checkbox("Use sample dataset (if no upload)", value=True)
    lang_target = st.selectbox("Response language", ["en", "hi", "bn", "mr", "ta", "te", "kn", "gu", "pa"], index=0)
    enable_translate = st.checkbox("Translate query/response", value=False)
    enable_tts = st.checkbox("Enable TTS (gTTS)", value=True)
    top_k = st.slider("Top K results (FAISS)", min_value=1, max_value=5, value=1)
    st.markdown("---")
    st.markdown("Voice input: upload a short audio file (wav/mp3) to transcribe.")
    audio_upload = st.file_uploader("Upload voice file for transcription", type=["wav","mp3","m4a"], accept_multiple_files=False)

with col1:
    st.header("Ask about a temple")
    query = st.text_input("Enter your question (example: 'Tell me about Kashi Vishwanath')")
    st.button("Ask", key="ask_button")

# ----------------------------- Dataset -----------------------------
sample_data = [
    {
        "name": "Kashi Vishwanath Temple",
        "location": "Varanasi, Uttar Pradesh",
        "description": "One of the twelve Jyotirlingas dedicated to Lord Shiva.",
        "history": "Believed to have been rebuilt several times, last reconstructed by Ahilyabai Holkar in 1780.",
        "architecture": "Nagara style with a golden spire.",
        "unique_features": "Golden domes and proximity to the Ganga river."
    },
    {
        "name": "Somnath Temple",
        "location": "Prabhas Patan, Gujarat",
        "description": "First among the twelve Jyotirlinga shrines of Lord Shiva.",
        "history": "Destroyed and rebuilt several times by invaders; current structure built in 1951.",
        "architecture": "Chalukya style with intricate carvings.",
        "unique_features": "Located near the Arabian Sea, known for its eternal shrine lamp."
    }
]

if uploaded is not None:
    try:
        temples = json.load(uploaded)
    except Exception as e:
        st.error(f"Failed to parse uploaded JSON: {e}")
        st.stop()
else:
    if use_sample:
        temples = sample_data
    else:
        try:
            temples = list(temples_collection.find({}, {"_id": 0}))
            if not temples:
                st.warning("No data found in MongoDB, using sample dataset.")
                temples = sample_data
        except Exception as e:
            st.error(f"MongoDB fetch failed: {e}")
            temples = sample_data

docs = build_documents(temples)
model = load_embedding_model()

@st.cache_data
def compute_embeddings(texts):
    return model.encode(texts, convert_to_numpy=True)

texts_for_embed = [d['raw'] for d in docs]
embeddings = compute_embeddings(texts_for_embed)

try:
    index = build_faiss_index(embeddings)
except Exception as e:
    st.error(f"FAISS error: {e}")
    st.stop()

# ----------------------------- Audio Input -----------------------------
transcribed_text = None
if audio_upload is not None:
    st.info("Transcribing uploaded audio — please wait...")
    ttext, terr = transcribe_audio_file(audio_upload)
    if terr:
        st.error(terr)
    else:
        transcribed_text = ttext
        st.success(f"Transcribed: {transcribed_text}")
        if not query:
            query = transcribed_text

# ----------------------------- Query handling -----------------------------
if query and st.session_state.get("ask_button_clicked", True):
    user_query = query.strip()

    if enable_translate and lang_target != "en":
        eng = translate_text(user_query, target_lang="en")
        st.write(f"Translated query -> en: {eng}")
        user_query_for_search = eng
    else:
        user_query_for_search = user_query

    q_emb = model.encode([user_query_for_search])
    D, I = index.search(np.array(q_emb).astype("float32"), top_k)

    results = []
    for score, idx in zip(D[0], I[0]):
        r = docs[idx]
        results.append({"distance": float(score), "doc": r})

    st.header("Results")
    for res in results:
        r = res["doc"]
        dist = res["distance"]
        st.subheader(r.get("name", ""))
        st.markdown(f"**Location:** {r.get('location', '')}")
        st.markdown(f"**Description:** {r.get('description', '')}")
        st.markdown(f"**History:** {r.get('history', '')}")
        st.markdown(f"**Architecture:** {r.get('architecture', '')}")
        st.markdown(f"**Unique features:** {r.get('unique_features', '')}")
        st.caption(f"L2 Distance: {dist:.4f} (lower = more similar)")

        if enable_translate and lang_target != "en":
            structured_text = (
                f"Name: {r.get('name', '')}\n"
                f"Location: {r.get('location', '')}\n"
                f"Description: {r.get('description', '')}\n"
                f"History: {r.get('history', '')}\n"
                f"Architecture: {r.get('architecture', '')}\n"
                f"Unique features: {r.get('unique_features', '')}"
            )
            translated = translate_text(structured_text, target_lang=lang_target)
            st.markdown("**Translated response:**")
            st.write(translated)

        if enable_tts:
            try:
                resp_text = f"{r.get('name', '')}. {r.get('description', '')}"
                lang_for_tts = lang_target if lang_target in ["en","hi","bn","mr","ta","te","kn","gu","pa"] else "en"
                audio_bytes = tts_and_audio_bytes(resp_text, lang=lang_for_tts)
                st.audio(audio_bytes, format="audio/mp3")
                b64 = base64.b64encode(audio_bytes).decode()
                href = f"<a href='data:audio/mp3;base64

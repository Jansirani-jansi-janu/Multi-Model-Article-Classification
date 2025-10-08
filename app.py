# app.py
import streamlit as st
import joblib
import numpy as np
import base64
from utils import clean_text

# Keras / TensorFlow imports
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# Hugging Face Transformers imports
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import os

# ---------------- ML MODEL ----------------
ml_model = joblib.load("models/ml_model.pkl")
vectorizer = joblib.load("models/tfidf.pkl")

# ---------------- DL MODEL ----------------
with open("models/tokenizer.json", "r", encoding="utf-8") as f:
    json_string = f.read()
    tokenizer = tokenizer_from_json(json_string)

try:
    dl_model = tf.keras.models.load_model("models/dl_model.h5")
except Exception as e:
    st.error(f"Error loading DL model: {e}")
    dl_model = None

# ---------------- TRANSFORMER MODEL ----------------
transformer_model_name = "models/transformer_model"
transformer_model = None
transformer_tokenizer = None

if os.path.exists(transformer_model_name):
    try:
        transformer_tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
        transformer_model = TFAutoModelForSequenceClassification.from_pretrained(
            transformer_model_name,
            num_labels=4,
            from_pt=True
        )
    except Exception as e:
        st.error(f"Error loading Transformer model: {e}")
else:
    st.error(f"Transformer model folder not found at {transformer_model_name}")


# ---------------- BACKGROUND IMAGE ----------------
def add_bg_from_local(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
            font-family: 'Segoe UI', sans-serif;
        }}

        /* Title */
        .app-title {{
            text-align: center;
            font-size: 50px;
            font-weight: bold;
            color: #ffffff;
            text-shadow: 3px 3px 6px #000000;
            margin-bottom: 20px;
        }}

        /* Header text */
        .app-header {{
            text-align: center;
            font-size: 24px;
            color: #ffcc00;
            margin-bottom: 30px;
            font-weight: bold;
            text-shadow: 1px 1px 3px #000000;
        }}

        /* Input labels */
        .label-bold {{
            font-size: 20px;
            font-weight: 700;
            color: #ffcc00;
            text-shadow: 1px 1px 2px #000;
            margin-top: 15px;
            margin-bottom: 5px;
            display: block;
        }}

        .subtext {{
            font-size: 16px;
            color: #cccccc;
            font-style: italic;
            margin-bottom: 15px;
            display: block;
        }}

        /* Dropdown (Selectbox) Styling */
        div[data-baseweb="select"] > div {{
            font-weight: bold;
            font-size: 18px;
            color: #1a1a1a;
        }}

        /* Button Styling */
        div.stButton > button:first-child {{
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            font-weight: bold;
            border-radius: 10px;
            padding: 10px 24px;
            border: none;
            cursor: pointer;
            transition: 0.3s;
        }}
        div.stButton > button:first-child:hover {{
            background-color: #45a049;
            transform: scale(1.05);
        }}

        /* Prediction Result Box */
        .prediction-box {{
            text-align: center;
            font-size: 22px;
            font-weight: bold;
            color: #ffffff;
            background-color: #ff5733;
            padding: 15px;
            border-radius: 12px;
            margin-top: 20px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# ✅ Use your image path
add_bg_from_local("C:/Users/Admin/Desktop/Jansi/Project/Multi model article classification/Evolution.png")

# ---------------- UI ----------------
st.markdown('<div class="app-title">Multi-Model Article Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="app-header">Classify news articles using ML, DL & Transformers</div>', unsafe_allow_html=True)

# Styled Labels
st.markdown('<span class="label-bold">Enter article text (title + description):</span>', unsafe_allow_html=True)
text = st.text_area("", height=120)


# Run Classification
if st.button("Classify"):
    if not text.strip():
        st.warning("Please enter text.")
    else:
        cleaned = clean_text(text)
        results = {}

        # Machine Learning
        X = vectorizer.transform([cleaned])
        results["Machine Learning"] = int(ml_model.predict(X)[0])

        # Deep Learning
        if dl_model is None:
            results["Deep Learning"] = "Model not loaded ❌"
        else:
            seq = tokenizer.texts_to_sequences([cleaned])
            seq = pad_sequences(seq, maxlen=200)
            pred_index = np.argmax(dl_model.predict(seq), axis=1)[0]
            results["Deep Learning"] = int(pred_index + 1)

        # Transformer
        if transformer_model is None or transformer_tokenizer is None:
            results["Transformer"] = "Model not loaded ❌"
        else:
            inputs = transformer_tokenizer(
                cleaned, return_tensors="tf", padding=True, truncation=True, max_length=200
            )
            logits = transformer_model(**inputs).logits
            pred_index = tf.argmax(logits, axis=1).numpy()[0]
            results["Transformer"] = int(pred_index + 1)

        # Label map
        label_map = {
            1: "World 🌍",
            2: "Sport ⚽",
            3: "Business 💼",
            4: "Sci/Tech 🔬"
        }

        # Show results
        for model, pred in results.items():
            if isinstance(pred, int):
                category = label_map.get(pred, pred)
                st.markdown(
                    f'<div class="prediction-box">{model} → Predicted Category: {category}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="prediction-box">{model} → {pred}</div>',
                    unsafe_allow_html=True
                )

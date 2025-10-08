import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from utils import clean_text

# Load dataset (skip header row, assign names)
df = pd.read_csv(
    "C:/Users/Admin/Desktop/Jansi/Project/Multi model article classification/Dataset/train.csv",
    skiprows=1,
    names=["label", "title", "desc"]
)

# ✅ Use FULL dataset (no sample here)
df["text"] = (df["title"].fillna("") + " " + df["desc"].fillna("")).apply(clean_text)

texts = df["text"].tolist()
labels = df["label"].astype(int).tolist()

# Tokenization
num_words = 20000    # increase vocab
maxlen = 250         # allow longer text
tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=maxlen)
y = np.array(labels) - 1   # Convert 1–4 → 0–3

# Train/Test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Build LSTM model
model = Sequential([
    Embedding(num_words, 128, input_length=maxlen),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(4, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Early stopping (stop when validation accuracy doesn't improve)
early_stop = EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)

# Train model
history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1
)

# Save model + tokenizer
os.makedirs("models", exist_ok=True)
model.save("models/dl_model.h5")

import json
tokenizer_json = tokenizer.to_json()
with open("models/tokenizer.json", "w", encoding="utf-8") as f:
    f.write(tokenizer_json)

print("✅ DL model trained on full dataset & saved in models/ folder")

# python
# train_transformer.py
import pandas as pd
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# 1. Load dataset (skip header row)
df = pd.read_csv(
    "C:/Users/Admin/Desktop/Jansi/Project/Multi model article classification/Dataset/train.csv",
    skiprows=1,
    names=["label", "title", "desc"]
)

# Combine title + description
df["text"] = df["title"].fillna("") + " " + df["desc"].fillna("")

# Hugging Face expects labels starting from 0
df["label"] = df["label"].astype(int) - 1  # convert 1–4 → 0–3

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df[["text", "label"]])

# 2. Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

dataset = dataset.map(tokenize_fn, batched=True)

# Train/test split (90/10)
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# 3. Model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=4
)

# 4. Training setup (latest transformers supports these args)
training_args = TrainingArguments(
    output_dir="models/transformer_model",
    eval_strategy="epoch",         # ✅ correct for your version
    save_strategy="epoch",         # ✅ already valid
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="logs",
    logging_steps=50,
)


# 5. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
)

# 6. Train
trainer.train()

# 7. Save model + tokenizer
model.save_pretrained("models/transformer_model")
tokenizer.save_pretrained("models/transformer_model")

print("✅ Transformer model and tokenizer saved in models/transformer_model/")
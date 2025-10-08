# utils.py
import re
import string
import nltk
from nltk.corpus import stopwords

# download stopwords the first time
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    """Clean text: lowercase, remove urls, html, punctuation, stopwords."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)   # remove urls
    text = re.sub(r"<.*?>", "", text)            # remove html
    text = re.sub(r"[^\x00-\x7F]+", "", text)    # remove emojis
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens)

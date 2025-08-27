# -----------------------------
# 2) Text cleaning
# -----------------------------
import nltk
import unicodedata
import string
from typing import List

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(
    text: str,
    lower: bool = True,
    remove_punct: bool = True,
    remove_stop: bool = True,
    lemmatize_tokens: bool = True,
    remove_numbers: bool = True,
    keep_only_alpha: bool = True
):
    text = unicodedata.normalize("NFKD", text)
    if lower:
        text = text.lower()
    if remove_punct:
        text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = nltk.word_tokenize(text)

    cleaned_tokens: List[str] = []
    for token in tokens:
        if remove_numbers and token.isdigit():
            continue
        if keep_only_alpha and not token.isalpha():
            continue
        if remove_stop and token in stop_words:
            continue
        if lemmatize_tokens:
            token = lemmatizer.lemmatize(token)
        cleaned_tokens.append(token)
    return " ".join(cleaned_tokens)
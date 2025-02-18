import re
import string
import polars as pl
import nltk
import spacy
from bs4 import BeautifulSoup
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from gensim.models.phrases import Phrases, Phraser

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

nlp = spacy.load("en_core_web_sm")
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
def clean_arxiv_abstract(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove display math (DOTALL to catch newlines)
    text = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)
    # Remove inline math expressions
    text = re.sub(r'\$.*?\$', '', text)
    # Remove LaTeX commands (e.g., \emph{...}, \textbf{...})
    text = re.sub(r'\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})?', '', text)
    # Remove citation markers like [1] or [1,2,3]
    text = re.sub(r'\[[^\]]*\]', '', text)
    # Collapse extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_text(text: str) -> list:
    """Tokenize a string into words."""
    if not isinstance(text, str):
        return []
    return word_tokenize(text)

def remove_stopwords(tokens: list) -> list:
    """Remove stopwords from a list of tokens."""
    return [token for token in tokens if token not in stop_words]

def lemmatize_tokens(tokens: list) -> list:
    """Lemmatize tokens using spaCy."""
    doc = nlp(" ".join(tokens))
    return [token.lemma_ for token in doc]

def stem_tokens(tokens: list) -> list:
    """Stem tokens using NLTK's PorterStemmer."""
    return [ps.stem(token) for token in tokens]

def named_entity_recognition(text: str) -> list:
    """Extract named entities from text using spaCy."""
    if not isinstance(text, str):
        return []
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def pos_tagging(text: str) -> list:
    """Extract POS tags from text using spaCy."""
    if not isinstance(text, str):
        return []
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]

def build_phraser(token_lists: list) -> Phraser:
    """Build a bigram phraser using Gensim's Phrases."""
    phrases = Phrases(token_lists, min_count=1, threshold=2)
    return Phraser(phrases)

def apply_phrases(tokens: list, phraser: Phraser) -> list:
    """Apply a pre-built phraser to a list of tokens."""
    return phraser[tokens]

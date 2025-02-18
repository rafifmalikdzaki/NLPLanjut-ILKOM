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

# -------------------------------
# Load ArXiv Dataset from a Parquet File and Sample 50,000 Documents
# -------------------------------
# Change "arxiv.parquet" to the path of your Parquet file.
df = pl.read_parquet("arxiv.parquet")
df = df.sample(n=50000, with_replacement=False)

# -------------------------------
# Process the Title Column
# -------------------------------
df = df.with_columns([
    pl.col("title").apply(clean_text).alias("title_clean"),
    pl.col("title").apply(lambda x: tokenize_text(clean_text(x))).alias("title_tokens")
])
df = df.with_columns([
    pl.col("title_tokens").apply(remove_stopwords).alias("title_tokens_no_stop"),
    pl.col("title_tokens_no_stop").apply(lemmatize_tokens).alias("title_lemmatized"),
    pl.col("title_tokens_no_stop").apply(stem_tokens).alias("title_stemmed"),
    pl.col("title_clean").apply(named_entity_recognition).alias("title_entities"),
    pl.col("title_clean").apply(pos_tagging).alias("title_pos_tags")
])

# -------------------------------
# Process the Abstract Column
# -------------------------------
df = df.with_columns([
    pl.col("abstract").apply(clean_arxiv_abstract).alias("abstract_clean"),
    pl.col("abstract").apply(lambda x: tokenize_text(clean_arxiv_abstract(x))).alias("abstract_tokens")
])
df = df.with_columns([
    pl.col("abstract_tokens").apply(remove_stopwords).alias("abstract_tokens_no_stop"),
    pl.col("abstract_tokens_no_stop").apply(lemmatize_tokens).alias("abstract_lemmatized"),
    pl.col("abstract_tokens_no_stop").apply(stem_tokens).alias("abstract_stemmed"),
    pl.col("abstract_clean").apply(named_entity_recognition).alias("abstract_entities"),
    pl.col("abstract_clean").apply(pos_tagging).alias("abstract_pos_tags")
])

# -------------------------------
# Phrase Detection
# -------------------------------
# Build phrasers from token lists for title and abstract
title_token_lists = df["title_tokens"].to_list()
abstract_token_lists = df["abstract_tokens"].to_list()
title_phraser = build_phraser(title_token_lists)
abstract_phraser = build_phraser(abstract_token_lists)

df = df.with_columns([
    pl.col("title_tokens").apply(lambda tokens: apply_phrases(tokens, title_phraser)).alias("title_phrases"),
    pl.col("abstract_tokens").apply(lambda tokens: apply_phrases(tokens, abstract_phraser)).alias("abstract_phrases")
])

# -------------------------------
# Save the Preprocessed Data
# -------------------------------
df.write_csv("arxiv_preprocessed.csv")
print("Preprocessing complete. Preprocessed data saved to 'arxiv_preprocessed.csv'.")

from rake_nltk import Rake
import nltk

NEEDED = [
    ("corpora/stopwords", "stopwords"),
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),  # NLTK ≥3.8 uses punkt_tab
]

def _ensure_nltk_data():
    for path, pkg in NEEDED:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg)

class RakeWrapper:
    def __init__(self, language: str = "english"):
        _ensure_nltk_data()
        self.rake = Rake(language=language)

    def extract(self, text: str, k: int = 5):
        self.rake.extract_keywords_from_text(text or "")
        phrases = self.rake.get_ranked_phrases_with_scores()
        return [{"text": p, "weight": round(float(s), 3)} for s, p in phrases[:k]]

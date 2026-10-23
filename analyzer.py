"""Analyze sentiment, readability, and TF-IDF keywords from text."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from textstat import flesch_reading_ease
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def analyze(text: str, top_k: int = 8) -> dict:
    if not text.strip():
        raise ValueError("Input text is empty.")
    words = re.findall(r"\b[\w'-]+\b", text.lower())
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=1000)
    try:
        matrix = vectorizer.fit_transform([text])
        scores = matrix.toarray()[0]
        terms = vectorizer.get_feature_names_out()
        keywords = sorted(zip(terms, scores), key=lambda item: item[1], reverse=True)[:top_k]
    except ValueError:  # e.g. text consists only of stop words
        keywords = []
    return {
        "sentiment": SentimentIntensityAnalyzer().polarity_scores(text),
        "statistics": {
            "characters": len(text), "words": len(words),
            "sentences": max(1, len(re.findall(r"[.!?]+", text))),
            "unique_words": len(set(words)), "flesch_reading_ease": round(flesch_reading_ease(text), 2),
        },
        "keywords": [{"term": term, "score": round(float(score), 4)} for term, score in keywords],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Text to analyze")
    source.add_argument("--file", type=Path, help="UTF-8 text file to analyze")
    parser.add_argument("--json", type=Path, help="Optional path to save the JSON report")
    args = parser.parse_args()
    text = args.text if args.text is not None else args.file.read_text(encoding="utf-8")
    result = analyze(text)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.json:
        args.json.write_text(rendered + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()

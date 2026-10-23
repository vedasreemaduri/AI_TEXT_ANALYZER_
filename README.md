# AI Text Analyzer

A small, extensible NLP project that analyzes sentiment, readability, and frequent terms in a text file or pasted text. It is designed as a clean portfolio starter with reproducible local analysis.

## Features
- VADER sentiment scores (positive, neutral, negative, compound)
- Word, sentence, and character counts plus Flesch reading ease
- TF-IDF keyword extraction
- JSON output for downstream use

## Setup
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python src/analyzer.py --text "The product is simple and works beautifully." 
python src/analyzer.py --file sample.txt --json report.json
```

## Next steps
Add a web interface, multilingual models, topic classification, and a labeled evaluation set. Sentiment scores are estimates and can miss sarcasm or context.

# bido_NER
*A template for bidos named NER service* 
Install

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows PowerShell

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Basic run

```bash
python ner_scraper.py https://example.com
```

Crawl up to 50 pages, two link levels deep

```bash
python ner_scraper.py https://example.com --max-pages 50 --depth 2
```

Keep selected NER labels

```bash
python ner_scraper.py https://example.com \
  --labels PERSON ORG GPE LOC EVENT
```

Output

By default, files are written under ner_output/:

• ner_results.json — page results plus aggregate entities
• entities.csv — deduplicated entity summary
• entity_occurrences.csv — each extracted entity occurrence

Use --include-text if you also want the extracted page text stored in JSON.
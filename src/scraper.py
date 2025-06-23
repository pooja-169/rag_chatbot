# src/scraper.py

from newspaper import Article

def extract_text_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        return f"[Scraper Error] {str(e)}"

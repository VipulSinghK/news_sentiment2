import os
import logging
from flask import Flask, render_template, request, jsonify
import newspaper
from newspaper import Article
import spacy
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from textblob import TextBlob
import sqlite3
from datetime import datetime
import uuid
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Lazy-load NLP models
nlp = None
sentiment_analyzer = None
emotion_analyzer = None
bias_tokenizer = None
bias_model = None

def get_nlp():
    global nlp
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    return nlp

def get_sentiment_analyzer():
    global sentiment_analyzer
    if sentiment_analyzer is None:
        sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", revision="714eb0f")
    return sentiment_analyzer

def get_emotion_analyzer():
    global emotion_analyzer
    if emotion_analyzer is None:
        emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
    return emotion_analyzer

def get_bias_model():
    global bias_tokenizer, bias_model
    if bias_model is None:
        bias_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        bias_tokenizer = AutoTokenizer.from_pretrained(bias_model_name)
        bias_model = AutoModelForSequenceClassification.from_pretrained(bias_model_name)
    return bias_tokenizer, bias_model

# Map BERT sentiment output to political bias
def classify_bias(text):
    tokenizer, model = get_bias_model()
    inputs = tokenizer(text[:512], return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    if predicted_class <= 1:
        return "Left"
    elif predicted_class == 2:
        return "Neutral"
    else:
        return "Right"

# Summarize article using spaCy
def summarize_article(text, max_sentences=3):
    doc = get_nlp()(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    ranked_sentences = sorted(
        sentences,
        key=lambda s: len(get_nlp()(s).ents),
        reverse=True
    )
    return " ".join(ranked_sentences[:max_sentences])

# Simplified date extraction
def extract_publish_date(article):
    publish_date = article.publish_date
    if publish_date:
        return publish_date.strftime("%Y-%m-%d")
    return datetime.now().strftime("%Y-%m-%d")

# Get database path
def get_db_path():
    db_path = os.getenv("DB_PATH", "news.db")
    directory = os.path.dirname(db_path)
    if directory:  # Only create directory if it's non-empty
        os.makedirs(directory, exist_ok=True)
    return db_path

# Initialize database
def init_db():
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS articles")
    c.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id TEXT PRIMARY KEY,
            url TEXT UNIQUE,
            title TEXT,
            summary TEXT,
            bias TEXT,
            sentiment TEXT,
            emotion TEXT,
            date TEXT,
            category TEXT
        )
    """)
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")

# Process a single article
def process_single_article(url, category=None):
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute("SELECT id FROM articles WHERE url = ?", (url,))
    existing_article = c.fetchone()
    if existing_article:
        c.execute("SELECT * FROM articles WHERE url = ?", (url,))
        row = c.fetchone()
        conn.close()
        return {
            "id": row[0],
            "url": row[1],
            "title": row[2],
            "summary": row[3],
            "bias": row[4],
            "sentiment": row[5],
            "emotion": row[6],
            "date": row[7],
            "category": row[8]
        }
    
    try:
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()
    except Exception as e:
        conn.close()
        logger.error(f"Failed to process article {url}: {str(e)}")
        raise Exception(f"Failed to process article {url}: {str(e)}")
    
    summary = summarize_article(article.text)
    bias = classify_bias(article.text)
    sentiment = get_sentiment_analyzer()(article.text[:512])[0]['label']
    emotions = get_emotion_analyzer()(article.text[:512])[0]
    primary_emotion = max(emotions, key=lambda x: x['score'])['label']
    
    if not category:
        tech_keywords = ["technology", "tech", "software", "AI", "cybersecurity"]
        politics_keywords = ["politics", "election", "government", "policy"]
        if any(kw in article.text.lower() for kw in tech_keywords):
            category = "Technology"
        elif any(kw in article.text.lower() for kw in politics_keywords):
            category = "Politics"
        else:
            category = "General"
    
    article_date = extract_publish_date(article)
    
    article_id = str(uuid.uuid4())
    try:
        c.execute("""
            INSERT INTO articles (id, url, title, summary, bias, sentiment, emotion, date, category)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (article_id, url, article.title, summary, bias, sentiment, primary_emotion, article_date, category))
        conn.commit()
        logger.info(f"Article saved: {url}")
    except Exception as e:
        conn.close()
        logger.error(f"Failed to save article {url} to database: {str(e)}")
        raise Exception(f"Failed to save article {url} to database: {str(e)}")
    
    conn.close()
    
    return {
        "id": article_id,
        "url": url,
        "title": article.title,
        "summary": summary,
        "bias": bias,
        "sentiment": sentiment,
        "emotion": primary_emotion,
        "date": article_date,
        "category": category
    }

# Process multiple articles from a news source URL
def process_articles_from_source(source_url, category=None, max_articles=30, batch_size=10):
    try:
        source = newspaper.build(source_url, memoize_articles=False)
    except Exception as e:
        logger.error(f"Failed to build news source {source_url}: {str(e)}")
        raise Exception(f"Failed to build news source {source_url}: {str(e)}")
    
    articles_data = []
    batch = []
    for article in source.articles[:max_articles]:
        batch.append(article.url)
        if len(batch) >= batch_size or len(batch) + len(articles_data) >= max_articles:
            for url in batch:
                try:
                    article_data = process_single_article(url, category)
                    articles_data.append(article_data)
                except Exception as e:
                    logger.warning(f"Skipping article {url}: {str(e)}")
                    continue
            batch = []  # Clear batch after processing
    
    # Process any remaining articles
    for url in batch:
        try:
            article_data = process_single_article(url, category)
            articles_data.append(article_data)
        except Exception as e:
            logger.warning(f"Skipping article {url}: {str(e)}")
            continue
    
    logger.info(f"Processed {len(articles_data)} articles from {source_url}")
    return articles_data

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/fetch_news", methods=["POST"])
def fetch_news():
    data = request.json
    url = data.get("url")
    category = data.get("category")
    try:
        articles_data = process_articles_from_source(url, category)
        logger.info(f"Fetched {len(articles_data)} articles from {url}")
        return jsonify({"status": "success", "articles": articles_data})
    except Exception as e:
        logger.error(f"Error fetching news: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

@app.route("/api/get_articles")
def get_articles():
    sentiment = request.args.get("sentiment")
    category = request.args.get("category")
    
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    query = "SELECT * FROM articles"
    conditions = []
    params = []
    
    if sentiment:
        conditions.append("sentiment = ?")
        params.append(sentiment)
    if category:
        conditions.append("category = ?")
        params.append(category)
    
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY date DESC"
    
    c.execute(query, params)
    articles = [
        {
            "id": row[0],
            "url": row[1],
            "title": row[2],
            "summary": row[3],
            "bias": row[4],
            "sentiment": row[5],
            "emotion": row[6],
            "date": row[7],
            "category": row[8]
        } for row in c.fetchall()
    ]
    conn.close()
    logger.info(f"Retrieved {len(articles)} articles from database")
    return jsonify(articles)

if __name__ == "__main__":
    init_db()
    if os.getenv("FLASK_ENV") == "production":
        logger.info("Starting News Aggregator in production mode")
        port = int(os.getenv("PORT", 8080))  # Default to 8080 for Render
        app.run(host="0.0.0.0", port=port)
    else:
        logger.info("Starting News Aggregator in development mode")
        port = int(os.getenv("PORT", 5000))  # Default to 5000 for local
        app.run(debug=True, host="0.0.0.0", port=port)
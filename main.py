from flask import Flask, render_template, request, jsonify
import newspaper
from newspaper import Article
import spacy
from transformers import pipeline
from textblob import TextBlob
import sqlite3
from datetime import datetime
import uuid
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

app = Flask(__name__)

# Load NLP models
nlp = spacy.load("en_core_web_sm")
sentiment_analyzer = pipeline("sentiment-analysis")
emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

# Load BERT model for bias detection
bias_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
bias_tokenizer = AutoTokenizer.from_pretrained(bias_model_name)
bias_model = AutoModelForSequenceClassification.from_pretrained(bias_model_name)

# Map BERT sentiment output to political bias (simplified mapping for demo)
def classify_bias(text):
    inputs = bias_tokenizer(text[:512], return_tensors="pt", truncation=True, padding=True)
    outputs = bias_model(**inputs)
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
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    ranked_sentences = sorted(
        sentences,
        key=lambda s: len(nlp(s).ents),
        reverse=True
    )
    return " ".join(ranked_sentences[:max_sentences])

# Simplified date extraction (original version)
def extract_publish_date(article):
    publish_date = article.publish_date
    if publish_date:
        return publish_date.strftime("%Y-%m-%d")
    return datetime.now().strftime("%Y-%m-%d")

# Initialize database
def init_db():
    conn = sqlite3.connect("news.db")
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

# Process a single article
def process_single_article(url, category=None):
    conn = sqlite3.connect("news.db")
    c = conn.cursor()
    c.execute("SELECT id FROM articles WHERE url = ?", (url,))
    existing_article = c.fetchone()
    if existing_article:
        conn.close()
        c.execute("SELECT * FROM articles WHERE url = ?", (url,))
        row = c.fetchone()
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
        raise Exception(f"Failed to process article {url}: {str(e)}")
    
    summary = summarize_article(article.text)
    bias = classify_bias(article.text)
    sentiment = sentiment_analyzer(article.text[:512])[0]['label']
    emotions = emotion_analyzer(article.text[:512])[0]
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
    except Exception as e:
        conn.close()
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
def process_articles_from_source(source_url, category=None, max_articles=40):
    try:
        source = newspaper.build(source_url, memoize_articles=False)
    except Exception as e:
        raise Exception(f"Failed to build news source {source_url}: {str(e)}")
    
    articles_data = []
    for article in source.articles[:max_articles]:
        try:
            article_data = process_single_article(article.url, category)
            articles_data.append(article_data)
        except Exception as e:
            print(f"Skipping article {article.url}: {str(e)}")
            continue
    
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
        return jsonify({"status": "success", "articles": articles_data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/api/get_articles")
def get_articles():
    sentiment = request.args.get("sentiment")
    category = request.args.get("category")
    
    conn = sqlite3.connect("news.db")
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
    return jsonify(articles)

if __name__ == "__main__":
    init_db()
    app.run(debug=True)

  

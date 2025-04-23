from flask import Flask, request, render_template
import joblib
import os

app = Flask(__name__)

# Load models
model = joblib.load("model/resume_scorer.pkl")
tfidf = joblib.load("model/tfidf_vectorizer.pkl")

def load_keywords(industry):
    """Load industry-specific keywords"""
    with open(f"data/industry_keywords/{industry}.txt") as f:
        return [line.strip() for line in f]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    # Get user input
    resume_text = request.form["resume_text"]
    job_target = request.form["job_target"]
    
    # TF-IDF transformation
    features = tfidf.transform([resume_text])
    
    # Predict score (0-100)
    score = model.predict_proba(features)[0][1] * 100
    
    # Keyword analysis
    keywords = load_keywords(job_target)
    missing_kws = [kw for kw in keywords if kw not in resume_text.lower()]
    
    return render_template(
        "result.html",
        score=f"{score:.1f}",
        missing_kws=missing_kws,
        job_target=job_target
    )

if __name__ == "__main__":
    app.run(debug=True)
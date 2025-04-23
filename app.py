from flask import Flask, render_template, request
import nltk
import json
import re
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

nltk.download('wordnet')
nltk.download('omw-1.4')

app = Flask(__name__)

# Load keywords from JSON
with open("data/industry_keywords.json", "r") as file:
    INDUSTRY_KEYWORDS = json.load(file)

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower())
    return synonyms

def keyword_match_score(resume_text, job_role):
    resume_words = set(re.findall(r'\w+', resume_text.lower()))
    keywords = INDUSTRY_KEYWORDS.get(job_role.lower(), [])

    total_keywords = len(keywords)
    if total_keywords == 0:
        return 0

    match_count = 0
    for kw in keywords:
        synonyms = get_synonyms(kw)
        if kw.lower() in resume_words or any(s in resume_words for s in synonyms):
            match_count += 1

    return match_count / total_keywords

def tfidf_score(resume_text, keywords):
    documents = [resume_text] + keywords
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(documents)
    scores = vectors[0].toarray()[0]
    return float(np.mean(scores))

def structure_analysis(text):
    feedback = []
    structure_score = 1.0

    if len(re.findall(r'\n\n', text)) < 2:
        feedback.append("Add clear section breaks using double newlines.")
        structure_score -= 0.2

    if len(re.findall(r'\.', text)) / max(len(text.split()), 1) > 0.25:
        feedback.append("Shorten long paragraphs for better readability.")
        structure_score -= 0.2

    if re.search(r'is|was|were|be|been|being', text.lower()):
        feedback.append("Reduce passive voice usage.")
        structure_score -= 0.1

    return structure_score, feedback

def evaluate_resume(resume_text, job_role):
    match_score = keyword_match_score(resume_text, job_role)
    structure_score, feedback = structure_analysis(resume_text)

    total_score = round((match_score * 0.7 + structure_score * 0.3) * 100, 2)
    return {
        "total_score": total_score,
        "match_score": round(match_score, 2),
        "structure_score": round(structure_score, 2),
        "feedback": feedback
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    result = {}
    if request.method == 'POST':
        resume_text = request.form['resume']
        job_role = request.form['job']
        result = evaluate_resume(resume_text, job_role)
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

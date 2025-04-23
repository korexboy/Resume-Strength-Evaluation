import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

data = {
    "text": [
        "Data scientist with 5 years of Python and Machine Learning experience",
        "Marketing specialist skilled in SEO and Google Analytics",
        "Software engineer proficient in Java and Spring Framework"
    ],
    "category": ["Data_Science", "Marketing", "Engineering"]
}
df = pd.DataFrame(data)

tfidf = TfidfVectorizer(
    max_features=1000,
    stop_words='english',
    ngram_range=(1, 2)  # Includes phrases like "machine learning"
)

X = tfidf.fit_transform(df['text'])


# Label resumes (1=Strong, 0=Weak)
df['label'] = df['category'].apply(
    lambda x: 1 if x in ["Data_Science", "Engineering"] else 0
)

model = RandomForestClassifier()
model.fit(X, df['label'])

# Save models
import joblib
joblib.dump(model, "model/resume_scorer.pkl")
joblib.dump(tfidf, "model/tfidf_vectorizer.pkl")
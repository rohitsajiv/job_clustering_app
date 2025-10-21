import os
from sklearn.cluster import KMeans
import joblib

MODEL_FILENAME = 'job_cluster_model.pkl'

def train_model(skill_matrix, tfidf, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(skill_matrix)
    joblib.dump((kmeans, tfidf), MODEL_FILENAME)  # Save both model and vectorizer
    return kmeans

def load_model():
    if not os.path.exists(MODEL_FILENAME):
        raise FileNotFoundError(f"Model file '{MODEL_FILENAME}' not found. Please train the model first.")
    return joblib.load(MODEL_FILENAME)  # returns (model, tfidf)

def classify_jobs(jobs_df, tfidf, model):
    skills = jobs_df['Skills'].tolist()
    X_new = tfidf.transform(skills)  
    clusters = model.predict(X_new)
    jobs_df['cluster'] = clusters
    return jobs_df


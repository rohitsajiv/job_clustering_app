import os
from sklearn.cluster import KMeans
import joblib

MODEL_FILENAME = 'job_cluster_model.pkl'

def train_model(skill_matrix, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(skill_matrix)
    joblib.dump(kmeans, MODEL_FILENAME)
    return kmeans

def load_model():
    if not os.path.exists(MODEL_FILENAME):
        raise FileNotFoundError(f"Model file '{MODEL_FILENAME}' not found. Please train the model first.")
    return joblib.load(MODEL_FILENAME)

def classify_jobs(jobs_df, tfidf, model):
    # Ensure column name matches exactly: 'Skills'
    skills = jobs_df['Skills'].tolist()
    X_new = tfidf.transform(skills)
    clusters = model.predict(X_new)
    jobs_df['cluster'] = clusters
    return jobs_df


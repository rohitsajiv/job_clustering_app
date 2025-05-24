from scraper import scrape_karkidi_jobs
from preprocess import preprocess_skills
from cluster import train_model, load_model, classify_jobs

def run_job_pipeline():
    jobs = scrape_karkidi_jobs()
    if jobs.empty:
        print("No jobs found.")
        return

    # Preprocess skills to get TF-IDF matrix and vectorizer
    skill_matrix, tfidf = preprocess_skills(jobs)

    try:
        model = load_model()
        print("Loaded existing clustering model.")
    except FileNotFoundError:
        print("Model file not found, training new model...")
        model = train_model(skill_matrix)
        print("Training complete and model saved.")

    # Classify jobs into clusters
    print("Available columns:", jobs.columns.tolist())

    clustered_jobs = classify_jobs(jobs, tfidf, model)

    # Print a summary of the results
    for _, job in jobs.iterrows():
        print(f"Title: {job['Title']}, Cluster: {job['cluster']}")


if __name__ == '__main__':
    run_job_pipeline()

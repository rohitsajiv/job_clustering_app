import streamlit as st
from scraper import scrape_karkidi_jobs
from preprocess import preprocess_skills
from cluster import train_model, load_model, classify_jobs

def run_job_pipeline():
    st.title("Karkidi Job Clustering App")

    # Scrape jobs
    with st.spinner("Scraping jobs..."):
        jobs = scrape_karkidi_jobs()
    
    if jobs.empty:
        st.warning("No jobs found.")
        return

    # Preprocess skills
    st.info("Preprocessing skills...")
    skill_matrix, tfidf = preprocess_skills(jobs)

    # Load or train model
    try:
        model, tfidf = load_model()  # load both model and TF-IDF
        st.success("Loaded existing clustering model and TF-IDF vectorizer.")
    except FileNotFoundError:
        st.warning("Model not found. Training a new one...")
        model = train_model(skill_matrix, tfidf)  # pass tfidf to save it too
        st.success("Training complete and model saved.")

    # Classify jobs
    st.info("Classifying jobs into clusters...")
    clustered_jobs = classify_jobs(jobs, tfidf, model)

    # Display the dataframe with clusters
    st.subheader("Clustered Job Listings")
    st.dataframe(clustered_jobs[['Title', 'cluster']])

    # Optional: display jobs by cluster
    cluster_id = st.selectbox("Select a cluster to view its jobs:", clustered_jobs['cluster'].unique())
    filtered = clustered_jobs[clustered_jobs['cluster'] == cluster_id]
    st.write(f"Jobs in Cluster {cluster_id}:")
    st.table(filtered[['Title', 'Company', 'Location']])

# Only run if executing as a Streamlit app
if __name__ == '__main__':
    run_job_pipeline()

from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_skills(job_data):
    # Access the skills column correctly
    skills = job_data['Skills'].tolist()
    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    skill_matrix = tfidf.fit_transform(skills)
    return skill_matrix, tfidf

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def recommend_role(resume_text):
    job_roles = [
        "Data Scientist with Python Machine Learning Statistics",
        "Software Engineer Java Development Spring",
        "Data Analyst SQL Excel Python",
        "AI ML Engineer Deep Learning Neural Networks",
        "Web Developer HTML CSS JavaScript"
    ]

    vectorizer = TfidfVectorizer(stop_words='english')

    vectors = vectorizer.fit_transform([resume_text] + job_roles)

    similarity_scores = cosine_similarity(vectors[0:1], vectors[1:])

    best_match_index = similarity_scores.argmax()

    return job_roles[best_match_index]


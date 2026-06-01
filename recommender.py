from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# JOB ROLES DATABASE
jobs = {
    'Data Scientist': 'python machine learning sql data analysis statistics numpy pandas',
    'Web Developer': 'html css javascript react nodejs frontend backend',
    'DevOps Engineer': 'aws docker kubernetes cloud automation linux git cicd',
    'Backend Developer': 'python java sql apis rest databases nodejs',
    'AI Engineer': 'python machine learning deep learning tensorflow neural networks',
    'Data Analyst': 'sql excel data visualization power bi tableau statistics',
    'Mobile Developer': 'flutter react native android ios swift kotlin',
    'Cybersecurity Analyst': 'networking security linux firewalls encryption ethical hacking',
}

# GET USER SKILLS
print("=== Tech Stack Recommender ===")
print("Enter 3 skills you know (press Enter after each):")
skill1 = input("Skill 1: ")
skill2 = input("Skill 2: ")
skill3 = input("Skill 3: ")

user_skills = f"{skill1} {skill2} {skill3}"

# BUILD THE RECOMMENDATION ENGINE
job_names = list(jobs.keys())
job_descriptions = list(jobs.values())

# Add user profile to the list for vectorizing
all_text = job_descriptions + [user_skills]

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_text)

# Cosine Similarity between user and all jobs
user_vector = tfidf_matrix[-1]
job_vectors = tfidf_matrix[:-1]
scores = cosine_similarity(user_vector, job_vectors)[0]

# Sort and get Top 3
ranked = sorted(zip(job_names, scores), key=lambda x: x[1], reverse=True)

# SHOW RESULTS
print("\n=== Top 3 Recommended Career Paths For You ===")
for i, (job, score) in enumerate(ranked[:3], 1):
    print(f"{i}. {job} — Match Score: {round(score * 100, 1)}%")
from recommender import recommend_role

resume_text = """
Python machine learning data analysis pandas numpy
"""

recommended_role = recommend_role(resume_text)

print("Recommended Job Role:")
print(recommended_role)

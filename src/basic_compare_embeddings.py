from sentence_transformers import SentenceTransformer, util
import numpy as np

model = SentenceTransformer('all-mpnet-base-v2')

sentences = ["What's the weather like in New York today?",  "Can you tell me the current weather in New York?"]
embeddings = model.encode(sentences)
cosine_scores = util.cos_sim(embeddings[0], embeddings[1])
print("Cosine Similarity:", cosine_scores.item())

def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

# Function to compute dot product similarity
def dot_product_similarity(vec1, vec2):
    return np.dot(vec1, vec2)


euclidean = euclidean_distance(embeddings[0], embeddings[1])
dot_product = dot_product_similarity(embeddings[0], embeddings[1])


print(f"Euclidean Distance: {euclidean}")
print(f"Dot Product Similarity: {dot_product}")
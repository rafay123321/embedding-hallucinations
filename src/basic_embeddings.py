from sentence_transformers import SentenceTransformer, util

sbert_model = SentenceTransformer('msmarco-distilbert-base-tas-b')

embedding = sbert_model.encode(
    "American pizza is one of the nation's greatest cultural exports",
    show_progress_bar=True,
    convert_to_tensor=True
)

print("Embedding dimensions:", embedding.shape[0])

print(embedding)
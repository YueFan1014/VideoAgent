from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def compute_cosine_similarity(target_embedding, embedding_list):
    target_embedding_tensor = target_embedding.reshape(1, -1)
    # Compute cosine similarity
    similarity_scores = cosine_similarity(target_embedding_tensor, embedding_list)
    return similarity_scores.reshape(-1)


def top_k_indices(scores, k):
    max_len = scores.shape[0]
    k = min(max_len, k)
    indices = np.argsort(scores)[-k:][::-1]
    return list(indices)
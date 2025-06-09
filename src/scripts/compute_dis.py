

import numpy as np


def euclidean_distance(a,b):
    return np.linalg.norm(a-b)

def squared_euclidean_distance(a, b):
    return np.sum((a - b) ** 2)

def cosine_similarity(a, b):
    
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    return dot_product / (norm_a * norm_b)
import faiss
import numpy as np

class VectorStore:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)
    
    def add(self, vectors):
        self.index.add(np.array(vectors))
    
    def search(self, query_vector, k):
        return self.index.search(np.array([query_vector]), k)
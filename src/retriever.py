class Retriever:
    def __init__(self, vectorstore, embedder):
        self.vectorstore = vectorstore
        self.embedder = embedder
    
    def retrieve(self, query, top_k):
        query_vector = self.embedder.embed(query)
        return self.vectorstore.search(query_vector, top_k)
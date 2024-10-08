import yaml
from embeddings import Embedder
from vectorstore import VectorStore
from retriever import Retriever
from generator import Generator

def load_config():
    with open('config.yml', 'r') as file:
        return yaml.safe_load(file)

def main():
    config = load_config()
    
    embedder = Embedder(config['embeddings']['model'])
    vectorstore = VectorStore(384)  # Adjust dimension as needed
    retriever = Retriever(vectorstore, embedder)
    generator = Generator(config['generator']['model'], config['generator']['max_tokens'])
    
    # Your main RAG logic here

if __name__ == "__main__":
    main()
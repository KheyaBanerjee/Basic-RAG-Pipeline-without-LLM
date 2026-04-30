from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# -----------------------------------
# Here are all the required functions
# -----------------------------------

# 1. Document loading
def load_documents(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    return text

# 2. Text Chunking / doc splitting
def split_documents(text, chunk_size=250, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap    
    return chunks

# 3. Embeddings
def embed_texts(model, texts):
    embeddings = model.encode(texts)
    return np.array(embeddings).astype("float32")

# 4. Search Vector Database
def search_vector_db(index, query_embedding, chunks, k=3):
    distances, indices = index.search(query_embedding, k)
    results = []
    for idx in indices[0]:
        results.append(chunks[idx])
    return results

# 5. LLM; this is a fake one, but any LLM can be used
def llm(prompt):
    # Here the answer is simulated manually
    return """
According to the refund policy, customers can request a full refund within 30 days if the product is unused and in its original packaging. 
Refunds after 30 days are generally not allowed, but exceptions may be made if the product is defective or damaged during shipping.
"""


# -----------------
# Main RAG Pipeline
# -----------------
if __name__ == "__main__":
    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # Load document
    documents = load_documents("company_policy.txt")
    # Split into chunks
    chunks = split_documents(documents)
    # Create embedding
    embeddings = embed_texts(model, chunks)
    # Create FAISS vector database
    dimension = embeddings.shape[1]
    vector_db = faiss.IndexFlatL2(dimension)
    # Add embeddings to vector database
    vector_db.add(embeddings)
    # User query
    question = "Can a customer get a refund after 30 days?"
    # Embed the question
    query_embedding = embed_texts(model, [question])
    # Search top 3 relevant chunks
    top_chunks = search_vector_db(vector_db, query_embedding, chunks, k=3)
    # Build prompt
    context = "\n\n".join(top_chunks)
    prompt = f"""
Answer using only this context:

{context}

Question: {question}
"""
    # Get answer from LLM
    answer = llm(prompt)
    print("Retrieved Context:")
    print(context)
    print("\nFinal Answer:")
    print(answer)
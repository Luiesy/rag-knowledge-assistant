import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
embedding_model = SentenceTransformer(r"C:\Users\xluzz\.cache\modelscope\models\BAAI--bge-small-zh-v1.5\snapshots\master")
BASE_DIR = Path(__file__).resolve().parent.parent
chroma_client = chromadb.PersistentClient(path=str(BASE_DIR / "chroma_clean_db"))
collection = chroma_client.get_collection(name="test")
def knowledge_skill(question):
    query_vector = embedding_model.encode([question]).tolist()
    result = collection.query(query_embeddings=query_vector,n_results=3)
    distance = result["distances"][0][0]
    if distance >= 0.8:
        return "知识库中没有相关信息"
    context = "\n".join(result["documents"][0])
    return context
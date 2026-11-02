import chromadb
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(r"C:\Users\xluzz\.cache\modelscope\models\BAAI--bge-small-zh-v1.5\snapshots\master")
client=chromadb.PersistentClient(path="./chroma_clean_db")
collection=client.get_or_create_collection(name="test")
with open("knowledge.txt","r",encoding="utf-8")as f:text=f.read()
documents=text.split("。")
documents=[doc for doc in documents if doc]
vectors=model.encode(documents).tolist()
collection.upsert(ids=[str(i)for i in range(len(documents))],documents=documents,embeddings=vectors)
print("知识库建立完成！")
print("共加入",len(documents),"条知识。")

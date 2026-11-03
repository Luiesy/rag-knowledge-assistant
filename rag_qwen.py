import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
chroma_client = chromadb.PersistentClient(path="./chroma_clean_db")
collection = chroma_client.get_collection(name="test")
llm_client = OpenAI(base_url="http://localhost:1234/v1",api_key="lm-studio")
query = input("请输入问题:")
query_vector = model.encode([query]).tolist()
result = collection.query(query_embeddings=query_vector,n_results=3)
distance = result["distances"][0][0]
print("当前问题：", query)
print("最小距离：", distance)
threshold = 0.8
if distance >= threshold:
    print("知识库中没有相关信息")
else:
    context = "\n".join(result["documents"][0])
    print("检索到的知识：")
    print(context)
    response = llm_client.chat.completions.create(model="qwen3-4b",messages=[{"role": "system","content": "你是一个知识库问答助手，只能根据提供的知识库内容回答问题。不要使用自己的知识补充答案。"},{"role": "user","content": f"知识库内容：\n{context}\n\n问题：{query}"}],max_tokens=200)
    print("AI回答：")
    print(response.choices[0].message.content)

from fastapi import FastAPI,UploadFile,File
from fastapi.middleware.cors import CORSMiddleware
import hashlib
from pydantic import BaseModel
from openai import OpenAI
from pypdf import PdfReader
import json
from skills.knowledge_skill import knowledge_skill
from skills.calculator_skill import calculator_skill
import chromadb
from sentence_transformers import SentenceTransformer
embedding_model=SentenceTransformer(r"C:\Users\xluzz\.cache\modelscope\models\BAAI--bge-small-zh-v1.5\snapshots\master")
chroma_client = chromadb.PersistentClient(path="./chroma_clean_db")
collection = chroma_client.get_or_create_collection(name="test")
uploaded_files=set()
llm_client = OpenAI(base_url="http://localhost:1234/v1",api_key="lm-studio")
app=FastAPI()
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
class Question(BaseModel):
    question:str
    role:str="Python老师"
client=OpenAI(base_url="http://localhost:1234/v1",api_key="lm-studio")
@app.get("/") 
def hello():
    return{"message":"你好，这是我的第一个RAG AI API!"}
@app.post("/ask")
def ask(data:Question):
    query_vector = embedding_model.encode([data.question]).tolist()
    result = collection.query(query_embeddings=query_vector,n_results=3)
    distance = result["distances"][0][0]
    threshold = 0.8
    if distance >= threshold:   return{"回答":"知识库中没有相关信息"}
    context="\n".join(result["documents"][0])
    response=llm_client.chat.completions.create(model="qwen3-4b",messages=[{"role": "system", "content": f"""你是一个{data.role}。请严格根据下面到的知识回答问题。如果知识中没有相关信息，请回答“知识库中没有相关信息”，不要使用你自己的知识进行回答。知识库:{context}"""},{"role": "user", "content": data.question+"/no_think"}],max_tokens=200)
    answer=response.choices[0].message.content
    print("最终答案:",repr(answer))
    return{"回答":str(answer)}
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")
    documents = text.split("。")
    documents = [doc for doc in documents if doc.strip()]
    vectors = embedding_model.encode(documents).tolist()
    collection.upsert(ids=[f"{file.filename}_{i}" for i in range(len(documents))],documents=documents,embeddings=vectors)
    return {"文件名": file.filename,"加入知识库": len(documents),"状态": "上传成功"}
    return{"回答":answer}
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    content = await file.read()
    file_hash=hashlib.md5(content).hexdigest()
    print("文件指纹:",file_hash)
    if file_hash in uploaded_files:
        return{"文件名":file.filename,"状态":"文件已经上传过了"}
    with open(file.filename, "wb") as f:f.write(content)
    reader = PdfReader(file.filename)
    text = ""
    for page in reader.pages:page_text = page.extract_text()
    if page_text:text += page_text + "\n"
    chunk_size=300
    overlap=50
    documents=[]
    start=0
    while start<len(text):
        end=start+chunk_size
        chunk=text[start:end]
        if chunk.strip():
            documents.append(chunk.strip())
        start+=chunk_size-overlap
    vectors = embedding_model.encode(documents).tolist()
    collection.upsert(ids=[f"{file.filename}_{i}" for i in range(len(documents))],documents=documents,embeddings=vectors)
    uploaded_files.add(file_hash)
    return {"文件名": file.filename,"加入知识库": len(documents),"状态": "PDF上传成功"}
class AgentQuestion(BaseModel):
    question: str
@app.post("/agent")
def agent_ask(data: AgentQuestion):
    messages = [{"role": "system","content": """你是一个严格遵守知识来源规则的Agent。规则：1. 当用户询问个人知识库中的信息时，必须先调用 knowledge_search 工具。2. 只有知识库检索结果能够支持答案时，才能依据检索结果回答。3. 如果知识库没有相关信息，必须明确回答“知识库中没有相关信息”，不得使用你自己的知识补充答案。4. 当用户要求数学计算时，必须调用 calculator 工具，不得心算代替。5. 如果任务同时需要知识检索和数学计算，必须调用两个工具，并根据工具结果整合回答。6. 工具调用完成后，再给出最终回答。"""},{"role": "user", "content": data.question}]
    tools = [{"type": "function","function": {"name": "knowledge_search","description": "查询个人知识库","parameters": {"type": "object","properties": {"question": {"type": "string"}},"required": ["question"]}}},{"type": "function","function": {"name": "calculator","description": "计算数学表达式","parameters": {"type": "object","properties": {"expression": {"type": "string"}},"required": ["expression"]}}}]
    for step in range(5):
        print(f"Agent执行第{step+1}轮")
        try:
            response = client.chat.completions.create(model="qwen3-4b",messages=messages,tools=tools,tool_choice="auto")
        except Exception as e:
            print("Agent模型请求失败:",e)
            return{"回答":"模型请求失败，请检查LM Studio是否启动、模型是否加载。"}
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))
        if not message.tool_calls:
            return {"回答": message.content}
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                print("工具参数解析失败：", e)
                messages.append({"role": "tool","tool_call_id": tool_call.id,"content": "工具参数格式错误，请重新生成合法的 JSON 参数。"})
                continue
            try:
                if tool_name == "knowledge_search":
                    if "question" not in arguments or not arguments["question"].strip():
                        result = "工具参数错误：缺少有效的 question"
                    else:
                        result = knowledge_skill(arguments["question"])
                elif tool_name == "calculator":
                    if "expression" not in arguments or not arguments["expression"].strip():
                        result = "工具参数错误：缺少有效的 expression"
                    else:
                        result = calculator_skill(arguments["expression"])
                else:
                    result = "未知工具"
            except Exception as e:
                result = f"工具执行失败：{e}"
        messages.append({"role": "tool","tool_call_id": tool_call.id,"content": str(result)})
    return {"回答": "Agent已达到最大执行轮数，请重新尝试。"}
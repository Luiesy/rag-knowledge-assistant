# 基于 RAG 的 AI 个人知识库问答系统

## 项目简介
本项目基于检索增强生成（RAG）技术，构建了一个个人知识库问答系统。通过上传 PDF 文档、文本向量化、语义检索与大语言模型生成回答，实现针对指定知识内容的问答，并通过知识约束机制减少模型脱离知识库内容回答的问题。

## 项目功能
- PDF 文档上传与文本提取
- 文本分块与向量化处理
- 基于 ChromaDB 的向量存储与语义检索
- 基于检索结果生成回答
- 支持角色选择与问答交互
- Agent 工具调用：知识库检索与数学计算
- 知识约束与无相关内容时的拒答机制

## 技术栈
- Python
- FastAPI
- ChromaDB
- Sentence Transformers
- BGE-small-zh-v1.5
- LM Studio
- Qwen3-4B
- OpenAI-compatible API
- HTML / JavaScript
- pypdf

## 项目结构
```text
rag-knowledge-assistant/
├── main.py
├── rag_qwen.py
├── build_knowledge.py
├── index.html
├── knowledge.txt
├── requirements.txt
└── skills/
    ├── __init__.py
    ├── calculator_skill.py
    ├── knowledge_skill.py
    └── skill_registry.py
```

## 本地运行

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 启动 LM Studio
在本地启动 LM Studio，加载 Qwen3-4B 模型，并开启本地 API 服务。

### 3. 构建知识库
```bash
python build_knowledge.py
```

### 4. 启动 FastAPI
```bash
uvicorn main:app --reload
```

### 5. 访问应用
浏览器打开：
http://127.0.0.1:8000

## 项目说明
本项目目前为本地运行版本，在线部署与公开演示功能仍在完善中。

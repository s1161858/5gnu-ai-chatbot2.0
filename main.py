#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py - 5GNU RAG Chatbot 后端服务
技术栈：FastAPI + LangChain + ChromaDB + DeepSeek API (流式 SSE)
"""

import os
import json
import logging
from pathlib import Path
from typing import List, AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv

from openai import AsyncOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ──────────────────────────────────────────────
# 基础配置
# ──────────────────────────────────────────────
dotenv_path = find_dotenv()
load_dotenv(dotenv_path, override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY", "").strip().strip('\'"')
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip().rstrip("/")
DEEPSEEK_MODEL    = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5").strip()
VECTOR_STORE_DIR  = os.getenv("VECTOR_STORE_DIR", "./vector_store").strip()
COLLECTION_NAME   = os.getenv("COLLECTION_NAME", "5gnu_knowledge").strip()

# 兜底 Key
if not DEEPSEEK_API_KEY:
    DEEPSEEK_API_KEY = "sk-73eaa93b6a334508a42408121d3a28d4"
    logger.warning("⚠️  .env 未读到 KEY，已使用内置备用 Key")

# Top-K 检索，不设分数阈值
RETRIEVER_K = 5

# ──────────────────────────────────────────────
# System Prompt（强制 5GNU 人设）
# ──────────────────────────────────────────────
BASE_SYSTEM_PROMPT = """你是 5GNU（5G新媒体）公司的专属智能 Agent，代号「星翼」🌟。
你必须始终使用热情、专业、富有科技感的语气与用户交流，善用 Emoji 让对话更生动。
无论用户问什么问题，你都始终是「5GNU 专属 AI」，绝不能自称为通用 AI 或其他助手。

【5GNU 公司知识库资料】
{context}
【知识库资料结束】

【核心行为规则】
1. 📌 公司专属问题（无人机、STEAM教育、AOPA考证、天地足球、LAEC低空经济学校中心、御空5G智能机场等）：
   → 必须严格依据上方知识库资料回答，不得编造数据。
2. 📌 若用户问的是公司相关问题，但知识库中找不到精确答案：
   → 诚实告知"我需要向团队确认这个细节"，并主动介绍公司其他核心业务（如5G无人机培训、AOPA考证、天地足球赛事等）。
3. 📌 对于编程辅导、数学、通用科技问题：
   → 以「5GNU AI 助手」身份自由发挥，提供专业解答。
4. 📌 任何情况下都必须保持 5GNU 专属 Agent 的身份人设，不得破坏。
5. 📌 回答语言：与用户一致（中文/英文/繁体中文均可）。"""

# 空 Context 保底内容（确保人设不丢失）
FALLBACK_CONTEXT = """（当前问题未在知识库中找到精确匹配，请根据以下公司基础信息作答）

【5GNU 公司核心业务简介】
• 5GNU（5G新媒体）是一家专注于 5G 技术应用与无人机教育的科技公司，总部位于香港。
• 核心业务：
  - STEAM 无人机教育：从小学积木无人机到专业 AOPA 证书（可转换香港 AR 牌），全阶段覆盖。
  - AOPA 考证中心：5GNU 是香港地区 AOPA 唯一指定考试中心。
  - 天地足球赛事：无人机足球（天上）+ 机器人足球（地上），结合 5G 远程控制技术。
  - LAEC 低空经济学校中心：集无人机足球、群飞表演、培训于一体的学校推广方案。
  - 御空 5G 智能机场：与天宇经纬合作的 5G 无人机智能机场系统解决方案。
  - 5G 直播与表演：承办香港维多利亚港国家烟花晚会的 5G 直播及大型无人机表演。"""

# ──────────────────────────────────────────────
# FastAPI 应用初始化
# ──────────────────────────────────────────────
app = FastAPI(
    title="5GNU 智能 Agent API",
    description="5GNU 公司专属 RAG 知识库问答服务",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path("./static")
if static_dir.exists() and static_dir.is_dir():
    app.mount("/static", StaticFiles(directory="static", html=True), name="static")
    logger.info("📁 静态文件目录已挂载：./static -> /static")

@app.get("/")
async def get_index():
    return FileResponse("static/index.html")

# ──────────────────────────────────────────────
# 全局对象
# ──────────────────────────────────────────────
embedding_func  = None
vectorstore     = None
retriever       = None
deepseek_client = None

@app.on_event("startup")
async def startup_event():
    global embedding_func, vectorstore, retriever, deepseek_client

    # 1. Embedding
    logger.info(f"🤖 加载 Embedding 模型：{EMBEDDING_MODEL}")
    try:
        embedding_func = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("   ✅ Embedding 模型加载成功")
    except Exception as e:
        logger.error(f"   ❌ Embedding 加载失败：{e}")
        raise

    # ── 新增：云端部署自动构建知识库 ──
    if not os.path.exists(VECTOR_STORE_DIR):
        logger.info("☁️  云端未检测到向量库，正在自动调用 ingest.py 构建...")
        try:
            import ingest
            ingest.main()
            logger.info("   ✅ 知识库自动构建成功")
        except Exception as e:
            logger.warning(f"   ⚠️  知识库构建失败：{e}，将继续启动服务")

    # 2. ChromaDB（Top-K 无阈值）
    logger.info(f"💾 连接 ChromaDB：{VECTOR_STORE_DIR} / 集合：{COLLECTION_NAME}")
    try:
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_func,
            persist_directory=VECTOR_STORE_DIR,
        )
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RETRIEVER_K},
        )
        count = vectorstore._collection.count()
        logger.info(f"   ✅ ChromaDB 连接成功，共 {count} 条记录，Top-K={RETRIEVER_K}")
    except Exception as e:
        logger.error(f"   ❌ ChromaDB 连接失败：{e}")
        raise

    # 3. DeepSeek Client
    masked_key = f"{DEEPSEEK_API_KEY[:4]}***{DEEPSEEK_API_KEY[-4:]}" if DEEPSEEK_API_KEY else "EMPTY"
    logger.info(f"🔑 初始化 DeepSeek | Base URL: {DEEPSEEK_BASE_URL} | Key: {masked_key}")
    deepseek_client = AsyncOpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )
    logger.info("🚀 5GNU 智能 Agent 服务启动成功！访问 http://localhost:8000")

# ──────────────────────────────────────────────
# 数据模型
# ──────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

# ──────────────────────────────────────────────
# RAG 检索函数
# ──────────────────────────────────────────────
def retrieve_context(query: str) -> str:
    if retriever is None:
        return ""
    try:
        docs = retriever.invoke(query)
        if not docs:
            return ""
        context_parts = []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "知识库")
            source_short = Path(source).name if source != "知识库" else "知识库"
            context_parts.append(f"【片段{i} · 来源：{source_short}】\n{doc.page_content.strip()}")
        context = "\n\n".join(context_parts)
        logger.info(f"🔍 RAG 命中 {len(docs)} 段文档（query: {query[:30]}...）")
        return context
    except Exception as e:
        logger.error(f"❌ RAG 检索异常：{e}")
        return ""

# ──────────────────────────────────────────────
# 流式 SSE 生成器
# ──────────────────────────────────────────────
async def stream_deepseek(messages_for_llm: list) -> AsyncGenerator[str, None]:
    try:
        stream = await deepseek_client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages_for_llm,
            stream=True,
            temperature=0.7,
            max_tokens=2048,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                data = json.dumps({"content": delta.content}, ensure_ascii=False)
                yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"❌ DeepSeek 调用异常：{e}")
        error_data = json.dumps({"error": str(e)}, ensure_ascii=False)
        yield f"data: {error_data}\n\n"
        yield "data: [DONE]\n\n"

# ──────────────────────────────────────────────
# 核心接口：POST /chat
# ──────────────────────────────────────────────
@app.post("/chat")
async def chat(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages 不能为空")

    latest_user_msg = next(
        (msg.content for msg in reversed(request.messages) if msg.role == "user"), ""
    )

    logger.info(f"💬 收到提问：{latest_user_msg[:60]}...")

    context = retrieve_context(latest_user_msg)

    if context:
        logger.info("   📚 已注入知识库上下文")
        final_context = context
    else:
        logger.info("   🛡️  使用保底人设上下文")
        final_context = FALLBACK_CONTEXT

    system_prompt = BASE_SYSTEM_PROMPT.format(context=final_context)

    messages_for_llm = [{"role": "system", "content": system_prompt}]
    for msg in request.messages:
        if msg.role != "system":
            messages_for_llm.append({"role": msg.role, "content": msg.content})

    return StreamingResponse(
        stream_deepseek(messages_for_llm),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

# ──────────────────────────────────────────────
# 健康检查
# ──────────────────────────────────────────────
@app.get("/health")
async def health_check():
    db_count = 0
    if vectorstore:
        try:
            db_count = vectorstore._collection.count()
        except Exception:
            pass
    return {
        "status": "ok",
        "service": "5GNU 智能 Agent v2.0",
        "model": DEEPSEEK_MODEL,
        "retriever": f"Top-K={RETRIEVER_K}（无阈值）",
        "knowledge_count": db_count,
    }

# ──────────────────────────────────────────────
# 启动入口
# ──────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )

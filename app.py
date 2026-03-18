import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 强制加载环境变量
load_dotenv(find_dotenv(), override=True)

# 页面配置
st.set_page_config(page_title="5GNU 专属智能 Agent", page_icon="⚡", layout="centered")

# CSS 美化
st.markdown("""
<style>
    .stChatFloatingInputContainer { bottom: 20px; }
    .stChatMessage { border-radius: 10px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ 5GNU 专属智能 Agent")
st.caption("5G新媒体 · 低空智联 | 关于 STEAM教育、AOPA考证、天地足球、御空5G机场，随便问我！")

# 环境变量与兜底 Key
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip().strip('\'"')
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip().rstrip("/")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5").strip()
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "./vector_store").strip()
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "5gnu_knowledge").strip()

if not DEEPSEEK_API_KEY:
    st.error("⚠️ 请在 Streamlit Secrets 或 .env 中配置 DEEPSEEK_API_KEY")
    st.stop()

# 侧边栏清理按钮
with st.sidebar:
    if st.button("🗑️ 清空对话历史"):
        st.session_session['messages'] = []
        st.rerun()
    st.markdown("---")
    st.write("欢迎来到 5GNU AI 助手！")

# 初始化 Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
    # 欢迎语
    st.session_state.messages.append({
        "role": "assistant",
        "content": "您好！我是 **5GNU（5G新媒体）** 的专属智能 Agent 🚀\n\n关于无人机 STEAM 教育、AOPA 考证、天地足球赛事、LAEC 中心，或者御空 5G 机场系统，请随时向我提问！"
    })

# 缓存加载向量库与模型（防止每次对话重新加载）
@st.cache_resource(show_spinner="正在连接知识库...")
def init_rag_system():
    # 注意：如果在 Streamlit Cloud，这里可能需要触发构建逻辑
    if not os.path.exists(VECTOR_STORE_DIR):
        import ingest
        ingest.main()
        
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=VECTOR_STORE_DIR,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    return retriever, client

try:
    retriever, client = init_rag_system()
except Exception as e:
    st.error(f"系统初始化失败: {str(e)}")
    st.stop()

# System Prompt 模板
BASE_SYSTEM_PROMPT = """你是 5GNU（5G新媒体）公司的专属智能 Agent，代号「星翼」🌟。
你必须始终使用热情、专业、富有科技感的语气与用户交流。
【知识库资料】
{context}

规则：严格依据上方资料回答公司业务问题。若找不到，主动介绍公司其他业务。编程通用问题自由发挥。"""

FALLBACK_CONTEXT = "【5GNU 公司核心业务简介】：STEAM 无人机教育、AOPA 考证中心、天地足球赛事、LAEC 中心、御空 5G 智能机场。"

# 渲染历史对话
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 处理用户输入
if prompt := st.chat_input("请输入您的问题..."):
    # 1. 存入并显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 检索知识库
    docs = retriever.invoke(prompt)
    if docs:
        context = "\n\n".join([f"片段{i}:\n{doc.page_content}" for i, doc in enumerate(docs, 1)])
    else:
        context = FALLBACK_CONTEXT

    sys_prompt = BASE_SYSTEM_PROMPT.format(context=context)

    # 3. 组装请求消息
    messages_for_llm = [{"role": "system", "content": sys_prompt}]
    # 把去掉欢迎语之外的真实历史放进去
    for m in st.session_state.messages:
        if m["role"] != "system":
            messages_for_llm.append({"role": m["role"], "content": m["content"]})

    # 4. 流式调用与显示
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages_for_llm,
            stream=True,
            temperature=0.7,
        )
        response = st.write_stream(stream)
    
    st.session_state.messages.append({"role": "assistant", "content": response})

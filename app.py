"""
个人知识库 — Streamlit Web 界面
功能：文件上传、知识管理、智能问答
"""

import gc
import html
import json
import logging
import os
import sys
import time
import traceback
import uuid
import streamlit as st

# ==================== 日志配置 ====================
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("kb")
logger.setLevel(logging.DEBUG)

# 文件日志（每次启动覆盖）
_fh = logging.FileHandler(os.path.join(LOG_DIR, "app.log"), mode="w", encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_fh)

# 控制台日志
_sh = logging.StreamHandler()
_sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_sh)

# 把当前目录加入 path，确保能 import 本地模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from parsers import parse_file, chunk_text
from vector_store import VectorStore
from qa_chain import QAChain


# ==================== 设置持久化 ====================
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "settings.json")


def load_settings() -> dict:
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_settings(settings: dict):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def init_session_settings():
    if "_settings_loaded" not in st.session_state:
        saved = load_settings()
        st.session_state.api_key = saved.get("api_key", "")
        st.session_state.base_url = saved.get("base_url", "")
        st.session_state.model = saved.get("model", config.OPENAI_MODEL)
        st.session_state._settings_loaded = True


def persist_settings():
    save_settings({
        "api_key": st.session_state.get("api_key", ""),
        "base_url": st.session_state.get("base_url", ""),
        "model": st.session_state.get("model", config.OPENAI_MODEL),
    })


# ==================== 对话历史持久化 ====================
CHAT_HISTORY_FILE = os.path.join(os.path.dirname(__file__), "chat_history.json")


def load_chat_history() -> list:
    """加载对话历史"""
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 只保留最近 100 条消息，防止文件过大
                return data[-100:]
        except Exception:
            pass
    return []


def save_chat_history(messages: list):
    """保存对话历史"""
    try:
        # 保存时排除过大的字段（如嵌入的向量等）
        clean = []
        for m in messages:
            clean.append({"role": m["role"], "content": m["content"]})
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(clean, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ==================== 页面配置 ====================
st.set_page_config(
    page_title="知识库 · RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================== 自定义 CSS ====================
def inject_custom_css():
    st.markdown("""
    <style>
    /* ===== 全局字体和背景 ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ===== 侧边栏 ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a3e 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    [data-testid="stSidebar"] .stMarkdown h1 {
        font-size: 1.4rem !important;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    [data-testid="stSidebar"] .stMarkdown h3 {
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: rgba(255,255,255,0.45);
        font-weight: 600;
        margin-top: 1rem;
    }
    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 12px 16px;
    }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        font-size: 0.78rem !important;
        color: rgba(255,255,255,0.5);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.08) !important;
    }
    [data-testid="stSidebar"] .stCaption {
        color: rgba(255,255,255,0.55) !important;
        font-size: 0.8rem;
    }

    /* ===== 指标卡片（主区域） ===== */
    .metric-card {
        background: linear-gradient(135deg, rgba(102,126,234,0.08), rgba(118,75,162,0.08));
        border: 1px solid rgba(102,126,234,0.15);
        border-radius: 16px;
        padding: 20px 24px;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102,126,234,0.15);
    }
    .metric-card .number {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    .metric-card .label {
        font-size: 0.8rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 4px;
    }

    /* ===== 欢迎页卡片 ===== */
    .step-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 24px;
        transition: all 0.2s;
        height: 100%;
    }
    .step-card:hover {
        border-color: rgba(102,126,234,0.3);
        background: rgba(102,126,234,0.04);
    }
    .step-card .step-icon {
        font-size: 2rem;
        margin-bottom: 12px;
    }
    .step-card .step-num {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        text-align: center;
        line-height: 28px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .step-card h4 {
        margin: 8px 0 6px;
        font-size: 1rem;
        font-weight: 600;
    }
    .step-card p {
        color: #999;
        font-size: 0.85rem;
        line-height: 1.5;
        margin: 0;
    }

    /* ===== 特性标签 ===== */
    .feature-tag {
        display: inline-block;
        background: rgba(102,126,234,0.1);
        border: 1px solid rgba(102,126,234,0.2);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.78rem;
        color: #667eea;
        margin: 3px 4px;
    }

    /* ===== 文件列表项 ===== */
    .file-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 14px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        margin-bottom: 6px;
        font-size: 0.85rem;
    }
    .file-item:hover {
        background: rgba(255,255,255,0.06);
    }

    /* ===== 欢迎页 Hero ===== */
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.3;
        margin-bottom: 8px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #888;
        line-height: 1.6;
        max-width: 600px;
    }

    /* ===== 格式支持卡片 ===== */
    .format-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        transition: all 0.2s;
    }
    .format-card:hover {
        border-color: rgba(102,126,234,0.3);
    }
    .format-card .format-icon {
        font-size: 1.8rem;
        margin-bottom: 8px;
    }
    .format-card .format-name {
        font-size: 0.85rem;
        font-weight: 600;
    }
    .format-card .format-ext {
        font-size: 0.75rem;
        color: #888;
    }

    /* ===== 聊天消息美化 ===== */
    [data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 16px 20px;
        margin-bottom: 8px;
        border: 1px solid rgba(255,255,255,0.06);
    }

    /* ===== 隐藏 Streamlit 默认元素 ===== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ===== 检索结果卡片 ===== */
    .retrieval-card {
        background: rgba(102,126,234,0.05);
        border-left: 3px solid #667eea;
        border-radius: 0 10px 10px 0;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .retrieval-card .source {
        font-size: 0.75rem;
        color: #667eea;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .retrieval-card .score {
        float: right;
        font-size: 0.7rem;
        background: rgba(102,126,234,0.15);
        padding: 2px 8px;
        border-radius: 10px;
        color: #667eea;
    }

    /* ===== 上传区域美化 ===== */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(102,126,234,0.3);
        border-radius: 16px;
        padding: 8px;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(102,126,234,0.6);
    }

    /* ===== Slider 美化 ===== */
    [data-testid="stSlider"] {
        padding: 0 4px;
    }

    /* ===== 聊天气泡美化 ===== */
    [data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 16px 20px;
        margin-bottom: 8px;
        border: 1px solid rgba(255,255,255,0.06);
    }
    [data-testid="stChatMessage"][data-testid-type="user"] {
        background: rgba(102,126,234,0.08);
        border: 1px solid rgba(102,126,234,0.15);
    }
    [data-testid="stChatMessage"][data-testid-type="assistant"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
    }

    /* ===== Chat Input 固定在底部 ===== */
    [data-testid="stChatInput"] {
        border-radius: 16px;
        border: 1px solid rgba(102,126,234,0.3);
        background: rgba(255,255,255,0.03);
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 1px rgba(102,126,234,0.3);
    }
    </style>

    <!-- 自动滚动到底部（新消息在底部） -->
    <script>
    function scrollToBottom() {
        window.parent.scrollTo({top: window.parent.document.body.scrollHeight, behavior: 'instant'});
    }
    // 页面加载后滚动到底部
    setTimeout(scrollToBottom, 300);
    setTimeout(scrollToBottom, 800);
    </script>
    """, unsafe_allow_html=True)


# ==================== 加载向量库 ====================
@st.cache_resource
def load_vector_store():
    """加载向量库（预加载模式，首次调用时加载所有模型，之后从缓存返回）"""
    with st.spinner("🧠 首次加载模型中，约需 5-10 秒..."):
        status = st.empty()

        def progress_callback(msg: str):
            status.caption(msg)

        try:
            vs = VectorStore(progress_callback=progress_callback)
            status.empty()
            return vs
        except Exception as e:
            st.error(f"向量库初始化失败: {e}")
            raise


def init_qa_chain():
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = QAChain()
    return st.session_state.qa_chain


# ==================== 侧边栏 ====================
def sidebar():
    init_session_settings()

    with st.sidebar:
        # Logo + 标题
        st.markdown("# 🧠 知识库")
        st.caption("基于 RAG 的智能问答系统")

        st.markdown("---")

        # 统计卡片
        if "stats" not in st.session_state:
            vs = load_vector_store()
            st.session_state.stats = vs.get_stats()
        stats = st.session_state.stats

        c1, c2 = st.columns(2)
        with c1:
            st.metric("📄 文件", stats["total_sources"])
        with c2:
            st.metric("🧩 片段", stats["total_chunks"])

        st.markdown("---")

        # API 设置（可折叠）
        with st.expander("🔑 API 设置", expanded=not bool(st.session_state.api_key)):
            api_key = st.text_input(
                "API Key",
                value=st.session_state.api_key,
                type="password",
                placeholder="sk-...",
            )
            st.session_state.api_key = api_key

            base_url = st.text_input(
                "Base URL",
                value=st.session_state.base_url,
                placeholder="留空使用 OpenAI 官方地址",
            )
            st.session_state.base_url = base_url

            model_name = st.text_input(
                "模型",
                value=st.session_state.model,
                placeholder="gpt-3.5-turbo",
            )
            st.session_state.model = model_name

            persist_settings()

        # 检索设置（可折叠）
        with st.expander("⚙️ 检索设置"):
            default_threshold = getattr(config, "SCORE_THRESHOLD", 0.3)
            score_threshold = st.slider(
                "相似度阈值",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get("score_threshold", default_threshold),
                step=0.05,
                help="低于此值的结果将被过滤",
            )
            st.session_state.score_threshold = score_threshold
            st.caption(f"当前阈值: {score_threshold:.2f} · 数值越高越严格")

        st.markdown("---")

        # 文件管理
        if stats["sources"]:
            st.markdown("##### 📂 知识来源")
            for src in stats["sources"]:
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.caption(f"📄 {src}")
                with col2:
                    if st.button("✕", key=f"del_{src}", help=f"删除 {src}"):
                        vs = load_vector_store()
                        deleted = vs.delete_by_source(src)
                        if "stats" in st.session_state:
                            del st.session_state.stats
                        st.rerun()

        # 底部信息
        st.markdown("---")
        st.caption("🧠 语义分块 · 混合检索 · Reranker 重排序")


# ==================== 文件上传 ====================
def _sanitize_filename(filename: str) -> str:
    basename = os.path.basename(filename)
    return f"{uuid.uuid4().hex}_{basename}"


MAX_FILE_SIZE_MB = 50


def page_upload():
    st.markdown("### 📤 添加知识")
    st.caption("支持文件上传、拖拽上传、直接粘贴文本")

    # 格式支持卡片
    formats = [
        ("📝", "TXT / MD", ".txt .md"),
        ("📑", "PDF", ".pdf"),
        ("📘", "Word", ".doc .docx"),
        ("📊", "Excel", ".xlsx"),
        ("📋", "粘贴文本", "直接输入"),
    ]
    cols = st.columns(5)
    for col, (icon, name, ext) in zip(cols, formats):
        with col:
            st.markdown(f"""
            <div class="format-card">
                <div class="format-icon">{icon}</div>
                <div class="format-name">{name}</div>
                <div class="format-ext">{ext}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # 文件上传区域
    uploaded_files = st.file_uploader(
        "拖拽文件到此处，或点击选择文件",
        type=["txt", "md", "markdown", "pdf", "doc", "docx", "xlsx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.markdown(f"**已选择 {len(uploaded_files)} 个文件：**")
        for f in uploaded_files:
            size_mb = f.size / (1024 * 1024)
            status = "⚠️ 过大" if size_mb > MAX_FILE_SIZE_MB else "✅ 就绪"
            st.caption(f"  {status} · {f.name} ({size_mb:.1f} MB)")

        st.markdown("")
        if st.button("🚀 开始入库", type="primary", use_container_width=True):
            vs = load_vector_store()
            total = len(uploaded_files)
            success_count = 0
            fail_count = 0
            skip_count = 0
            overall_start = time.time()

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, file in enumerate(uploaded_files):
                file_start = time.time()
                file_size_mb = file.size / (1024 * 1024)
                logger.info(f"开始处理文件 [{i+1}/{total}]: {file.name} ({file_size_mb:.1f}MB)")

                if file_size_mb > MAX_FILE_SIZE_MB:
                    logger.warning(f"文件过大，跳过: {file.name} ({file_size_mb:.1f}MB)")
                    st.warning(f"⚠️ {file.name} ({file_size_mb:.1f}MB) 超过限制，跳过")
                    skip_count += 1
                    continue

                status_text.caption(f"⏳ 处理中: {file.name} ({i+1}/{total})")

                safe_name = _sanitize_filename(file.name)
                file_path = os.path.join(config.UPLOAD_DIR, safe_name)
                assert os.path.realpath(file_path).startswith(os.path.realpath(config.UPLOAD_DIR))
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                logger.info(f"文件已保存: {file_path}")

                try:
                    text = parse_file(file_path)
                except Exception as e:
                    logger.error(f"文件解析失败: {file.name}\n{traceback.format_exc()}")
                    st.error(f"❌ {file.name} 解析失败: {e}")
                    fail_count += 1
                    continue

                if not text.strip():
                    st.warning(f"⚠️ {file.name} 内容为空，跳过")
                    skip_count += 1
                    continue

                chunks = chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
                count = vs.add_documents(chunks, source=file.name, file_size=file.size)
                file_elapsed = time.time() - file_start
                st.success(f"✅ {file.name} → {count} 个片段 ({file_elapsed:.1f}s)")
                success_count += 1

                del text, chunks
                gc.collect()
                progress_bar.progress((i + 1) / total)

            overall_elapsed = time.time() - overall_start
            status_text.caption("✨ 全部完成！")

            # 摘要
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("总计", f"{total} 个")
            col2.metric("成功", f"{success_count} 个")
            col3.metric("失败", f"{fail_count} 个")
            col4.metric("耗时", f"{overall_elapsed:.1f}s")

            if success_count > 0:
                st.balloons()
            if "stats" in st.session_state:
                del st.session_state.stats
            st.rerun()

    # ===== 直接粘贴文本 =====
    st.markdown("---")
    st.markdown("##### 📋 或者直接粘贴文本")
    paste_text = st.text_area(
        "粘贴或输入文本内容",
        height=180,
        placeholder="在此粘贴或输入文本内容，然后点击下方按钮入库...",
        label_visibility="collapsed",
    )
    paste_name = st.text_input(
        "来源名称（可选）",
        placeholder="例如：读书笔记、会议记录...",
        label_visibility="visible",
    )

    if paste_text.strip():
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("📥 文本入库", type="primary", use_container_width=True):
                vs = load_vector_store()
                name = paste_name.strip() or "粘贴文本"
                chunks = chunk_text(paste_text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
                count = vs.add_documents(chunks, source=name)
                st.success(f"✅ 已入库 {count} 个知识片段（来源: {name}）")
                if "stats" in st.session_state:
                    del st.session_state.stats
                st.rerun()
        with col2:
            st.caption(f"已输入 {len(paste_text)} 个字符，预计分为约 {len(paste_text) // config.CHUNK_SIZE + 1} 个片段")


# ==================== 问答页面 ====================
def page_qa():
    if not st.session_state.get("api_key"):
        st.info("请先在左侧面板设置 API Key")
        return

    vs = load_vector_store()
    qa = init_qa_chain()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 输入框（放在最前面，先获取用户输入）
    user_input = st.chat_input("输入你的问题...")

    # 显示历史消息（按时间顺序，旧的在上，新的在下）
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 处理新消息
    if user_input:
        # 用户消息
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # AI 回答
        with st.chat_message("assistant"):
            with st.status("🧠 思考中...", expanded=True) as status:
                st.write("🔍 正在检索知识库...")
                logger.info(f"用户提问: {user_input}")
                logger.info(f"知识库文档数: {vs.collection.count()}, BM25文档数: {len(vs._bm25_docs)}")

                results = vs.search_hybrid(user_input, top_k=config.TOP_K)
                logger.info(f"search_hybrid 返回 {len(results)} 条结果")
                if results:
                    logger.info(f"最高分: {results[0]['score']:.4f}, 最低分: {results[-1]['score']:.4f}")
                else:
                    logger.warning("search_hybrid 返回空! 尝试纯向量检索...")
                    vec = vs.search(user_input, top_k=5, score_threshold=None)
                    logger.info(f"纯向量检索返回 {len(vec)} 条")

            threshold = st.session_state.get("score_threshold", getattr(config, "SCORE_THRESHOLD", 0.3))
            if results:
                results = [r for r in results if r["score"] >= threshold]

            # 检索结果展示
            if not results:
                st.write("📭 知识库未命中，使用通用能力回答")
                status.update(label="✅ 检索完成（未命中）", state="complete", expanded=False)
            else:
                top_score = max(r["score"] for r in results)
                st.write(f"🔍 检索命中 {len(results)} 个片段 · 最高相似度 {top_score:.0%}")
                status.update(label="✅ 检索完成", state="complete", expanded=False)

                with st.expander("查看命中的知识片段"):
                    for i, r in enumerate(results, 1):
                        sc = r["score"] * 100
                        color = "#22c55e" if sc > 60 else "#eab308" if sc > 40 else "#ef4444"
                        safe_source = html.escape(r['source'])
                        safe_text = html.escape(r['text'][:200])
                        st.markdown(f"""
                        <div class="retrieval-card">
                            <span class="score" style="background:{color}22; color:{color};">#{i} · {sc:.0f}%</span>
                            <div class="source">📄 {safe_source}</div>
                            <div style="font-size:0.82rem; color:#aaa; margin-top:4px;">{safe_text}{'...' if len(r['text']) > 200 else ''}</div>
                        </div>
                        """, unsafe_allow_html=True)

            # 流式生成回答
            history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
            sources_holder = {"sources": []}

            def stream():
                gen = qa.answer_stream(user_input, results, chat_history=history) if results else qa.answer_simple_stream(user_input)
                for chunk in gen:
                    if isinstance(chunk, dict) and "sources" in chunk:
                        sources_holder["sources"] = chunk["sources"]
                    else:
                        yield chunk

            answer = st.write_stream(stream())

            if sources_holder["sources"]:
                with st.expander(f"📎 参考来源"):
                    for src in sources_holder["sources"]:
                        st.caption(f"📄 {src}")

            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources_holder["sources"]})


# ==================== 欢迎页 ====================
def _show_welcome():
    # Hero
    st.markdown("""
    <div style="text-align:center; padding:20px 0 10px;">
        <div class="hero-title">🧠 个人智能知识库</div>
        <div class="hero-subtitle">
            基于 RAG（检索增强生成）技术，上传文档即可用自然语言提问。<br>
            本地向量化 · 混合检索 · Reranker 精排 · 流式回答
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 特性标签
    st.markdown("""
    <div style="text-align:center; margin:16px 0 24px;">
        <span class="feature-tag">📄 多格式支持</span>
        <span class="feature-tag">🔍 语义检索</span>
        <span class="feature-tag">🏷️ BM25 关键词</span>
        <span class="feature-tag">📊 Reranker 精排</span>
        <span class="feature-tag">💬 多轮对话</span>
        <span class="feature-tag">⚡ 流式输出</span>
    </div>
    """, unsafe_allow_html=True)

    # 步骤卡片
    st.markdown("")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="step-card">
            <div class="step-num">1</div>
            <h4>设置 API Key</h4>
            <p>在左侧面板填写 OpenAI 或兼容 API 的 Key，用于调用大模型生成回答</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="step-card">
            <div class="step-num">2</div>
            <h4>上传文档</h4>
            <p>支持 TXT、PDF、Word、Excel 等格式，系统自动解析、分块、向量化入库</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="step-card">
            <div class="step-num">3</div>
            <h4>开始提问</h4>
            <p>用自然语言提问，系统检索知识库并结合大模型生成准确、有来源的回答</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # 示例导入按钮
    example_path = os.path.join(os.path.dirname(__file__), "data", "示例知识.txt")
    if os.path.exists(example_path):
        col_a, col_b, col_c = st.columns([1, 1, 1])
        with col_b:
            if st.button("📚 一键导入示例知识", type="primary", use_container_width=True):
                vs = load_vector_store()
                with open(example_path, "r", encoding="utf-8") as f:
                    text = f.read()
                chunks = chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
                count = vs.add_documents(chunks, source="示例知识.txt")
                if "stats" in st.session_state:
                    del st.session_state.stats
                st.success(f"✅ 已导入 {count} 个知识片段！")
                time.sleep(1)
                st.rerun()


# ==================== 主入口 ====================
# ==================== 知识管理页面 ====================
def page_manage():
    vs = load_vector_store()
    stats = vs.get_stats()

    if stats["total_sources"] == 0:
        st.markdown("### 📂 知识管理")
        st.markdown("")
        st.markdown("""
        <div style="text-align:center; padding:40px 0; color:#888;">
            <div style="font-size:3rem; margin-bottom:16px;">📭</div>
            <div style="font-size:1.1rem; font-weight:600;">知识库为空</div>
            <div>请先在「上传文档」页面添加知识</div>
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown("### 📂 知识管理")
    st.caption(f"共 {stats['total_sources']} 个文件，{stats['total_chunks']} 个知识片段")

    # ===== 搜索和筛选栏 =====
    col1, col2, col3 = st.columns([4, 2, 1])
    with col1:
        keyword = st.text_input("搜索文件名", placeholder="输入关键词...", label_visibility="collapsed")
    with col2:
        categories = vs.get_categories()
        selected_cat = st.selectbox("分类", categories, label_visibility="collapsed")
    with col3:
        search_clicked = st.button("🔍 搜索", use_container_width=True)

    # 搜索逻辑（按回车或点按钮都生效）
    search_keyword = keyword if (keyword or search_clicked) else None

    # ===== 批量操作栏 =====
    files = vs.list_files(category=selected_cat, keyword=search_keyword)
    if not files:
        st.info("没有匹配的文件")
        return

    col_a, col_b, col_c, col_d = st.columns([1, 1, 1, 2])

    def _toggle_select_all():
        """全选/取消全选回调"""
        checked = st.session_state.select_all
        for f in files:
            key = f"chk_{f['source']}"
            st.session_state[key] = checked

    with col_a:
        select_all = st.checkbox("全选", key="select_all", on_change=_toggle_select_all)
    with col_b:
        if st.button("🗑️ 删除选中", type="secondary"):
            selected = st.session_state.get("selected_files", [])
            if selected:
                deleted = vs.batch_delete(selected)
                st.success(f"已删除 {len(selected)} 个文件（{deleted} 个片段）")
                if "stats" in st.session_state:
                    del st.session_state.stats
                st.rerun()
            else:
                st.warning("请先勾选要删除的文件")
    with col_c:
        if st.button("📥 导出清单", type="secondary"):
            import csv
            import io
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["文件名", "分类", "片段数", "上传时间", "文件大小"])
            for f in files:
                writer.writerow([
                    f["source"],
                    f.get("category", "未分类"),
                    f.get("chunk_count", 0),
                    f.get("upload_time", ""),
                    f"{f.get('file_size', 0) / 1024:.0f}KB" if f.get("file_size") else "",
                ])
            st.download_button(
                "下载 CSV",
                output.getvalue(),
                file_name="知识库清单.csv",
                mime="text/csv",
            )

    st.markdown("")

    # ===== 文件列表 =====
    selected_files = []
    for f in files:
        source = f["source"]
        category = f.get("category", "未分类")
        chunk_count = f.get("chunk_count", 0)
        upload_time = f.get("upload_time", "")
        file_size = f.get("file_size", 0)

        # 格式化文件大小
        if file_size > 1024 * 1024:
            size_str = f"{file_size / 1024 / 1024:.1f}MB"
        elif file_size > 1024:
            size_str = f"{file_size / 1024:.0f}KB"
        else:
            size_str = ""

        col_check, col_info, col_cat, col_action = st.columns([0.3, 3, 1.2, 0.8])

        with col_check:
            checked = st.checkbox("", key=f"chk_{source}", value=select_all, label_visibility="collapsed")
            if checked:
                selected_files.append(source)

        with col_info:
            # 文件名 + 详情
            detail_parts = [f"🧩 {chunk_count} 片段"]
            if upload_time:
                detail_parts.append(f"🕐 {upload_time}")
            if size_str:
                detail_parts.append(f"📦 {size_str}")
            st.markdown(f"**{source}**")
            st.caption(" · ".join(detail_parts))

        with col_cat:
            # 分类选择
            cat_options = ["未分类", "工作", "技术", "学习", "资料", "其他"]
            current_idx = cat_options.index(category) if category in cat_options else 0
            new_cat = st.selectbox(
                "分类", cat_options, index=current_idx,
                key=f"cat_{source}", label_visibility="collapsed"
            )
            if new_cat != category:
                vs.update_category(source, new_cat)
                st.rerun()

        with col_action:
            if st.button("🗑️", key=f"mgmt_del_{source}", help=f"删除 {source}"):
                deleted = vs.delete_by_source(source)
                st.success(f"已删除 {deleted} 个片段")
                if "stats" in st.session_state:
                    del st.session_state.stats
                st.rerun()

        st.markdown('<hr style="margin:4px 0; border-color:rgba(255,255,255,0.06);">', unsafe_allow_html=True)

    # 保存选中状态
    st.session_state.selected_files = selected_files

    # ===== 底部统计 =====
    st.markdown("")
    col1, col2, col3 = st.columns(3)
    total_chunks = sum(f.get("chunk_count", 0) for f in files)
    total_size = sum(f.get("file_size", 0) for f in files)
    col1.metric("筛选结果", f"{len(files)} 个文件")
    col2.metric("知识片段", f"{total_chunks} 个")
    col3.metric("总大小", f"{total_size / 1024 / 1024:.1f}MB" if total_size > 0 else "N/A")


def main():
    inject_custom_css()

    # ===== 首次加载时显示加载动画 =====
    if "app_ready" not in st.session_state:
        loading_placeholder = st.empty()
        loading_placeholder.markdown("""
        <div style="text-align:center; padding:120px 0;">
            <div style="font-size:3rem; margin-bottom:20px; animation: pulse 1.5s infinite;">🧠</div>
            <div style="font-size:1.3rem; font-weight:600; background: linear-gradient(135deg, #667eea, #764ba2);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom:12px;">
                知识库正在加载
            </div>
            <div style="color:#888; font-size:0.9rem;">首次加载模型需要约 10 秒，请稍候...</div>
            <div style="margin-top:24px;">
                <div style="width:200px; height:3px; background:rgba(255,255,255,0.1); border-radius:2px; margin:0 auto; overflow:hidden;">
                    <div style="width:60%; height:100%; background:linear-gradient(90deg, #667eea, #764ba2);
                        border-radius:2px; animation: loading_bar 1.5s ease-in-out infinite;"></div>
                </div>
            </div>
        </div>
        <style>
        @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.1); opacity: 0.8; }
        }
        @keyframes loading_bar {
            0% { transform: translateX(-100%); }
            50% { transform: translateX(50%); }
            100% { transform: translateX(200%); }
        }
        </style>
        """, unsafe_allow_html=True)

    sidebar()

    # 加载向量库（首次加载时 loading_placeholder 还在显示）
    has_api_key = bool(st.session_state.get("api_key"))
    if "stats" not in st.session_state:
        vs = load_vector_store()
        st.session_state.stats = vs.get_stats()
    has_knowledge = st.session_state.stats["total_chunks"] > 0

    # 加载完成，清除加载动画
    if "app_ready" not in st.session_state:
        loading_placeholder.empty()
        st.session_state.app_ready = True

    if not has_api_key and not has_knowledge:
        _show_welcome()
        st.markdown("---")

    # 自定义 Tab 栏（用 session_state 记住当前 Tab，不会因 widget 交互而跳转）
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "qa"

    tabs = [
        ("qa", "💬 智能问答"),
        ("upload", "📤 上传文档"),
        ("manage", "📂 知识管理"),
    ]
    tab_cols = st.columns(len(tabs))
    for col, (tab_id, tab_label) in zip(tab_cols, tabs):
        with col:
            is_active = st.session_state.active_tab == tab_id
            if st.button(
                tab_label,
                key=f"tab_{tab_id}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.active_tab = tab_id
                st.rerun()

    st.markdown("")

    # 根据当前 Tab 渲染对应页面
    if st.session_state.active_tab == "qa":
        page_qa()
    elif st.session_state.active_tab == "upload":
        page_upload()
    elif st.session_state.active_tab == "manage":
        page_manage()


if __name__ == "__main__":
    main()

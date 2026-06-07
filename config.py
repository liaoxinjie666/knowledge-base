"""
知识库配置文件
所有可调参数集中在这里管理
"""

import os

# ==================== Embedding 模型 ====================
# 使用 BAAI/bge-small-zh-v1.5，中文效果好，体积小（~90MB）
# 首次使用会从 HuggingFace 下载模型，国内用户建议设置 HF_MIRROR_URL 环境变量
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
EMBEDDING_DIMENSION = 512
# HuggingFace 镜像源地址，国内网络环境建议使用 https://hf-mirror.com
HF_MIRROR_URL = os.getenv("HF_MIRROR_URL", "https://hf-mirror.com")

# ==================== ChromaDB 向量数据库 ====================
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "knowledge_base"

# ==================== 文档分块参数 ====================
CHUNK_SIZE = 500       # 每块最大字符数
CHUNK_OVERLAP = 50     # 相邻块的重叠字符数（保证上下文连贯）

# ==================== 层级分块参数 ====================
PARENT_CHUNK_SIZE = 1500    # 父块最大字符数
CHILD_CHUNK_SIZE = 400      # 子块最大字符数（用于精确检索）

# ==================== RAG 检索增强参数 ====================
ENABLE_HYDE = True          # 启用 HyDE（Hypothetical Document Embeddings）
ENABLE_MULTI_QUERY = True   # 启用多查询扩展
MULTI_QUERY_COUNT = 3       # 多查询扩展生成的查询数量
ENABLE_SELF_RAG = True      # 启用 Self-RAG 自我检索增强生成

# ==================== OpenAI / LLM ====================
# 通过环境变量 OPENAI_API_KEY 设置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# 兼容第三方 API（如 DeepSeek、智谱等），留空则用官方地址
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", None)
# 模型名称
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# ==================== 检索参数 ====================
TOP_K = 5  # 检索返回的最相关文档片段数量
SCORE_THRESHOLD = 0.3  # 相似度低于此值的结果不返回

# ==================== 文件上传 ====================
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ==================== 支持的文件格式 ====================
SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".markdown",      # 文本 / Markdown
    ".pdf",                           # PDF
    ".doc", ".docx",                  # Word（含旧版 .doc）
    ".xlsx", ".xls",                  # Excel
}

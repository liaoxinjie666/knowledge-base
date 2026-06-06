# 📚 个人知识库

基于 RAG（检索增强生成）的本地知识库系统，支持多种文档格式，语义检索 + LLM 智能问答。

## 功能

- 📤 **多格式上传**：支持 TXT / Markdown / PDF / Word / Excel
- 🧠 **本地向量化**：使用 bge-small-zh-v1.5，离线运行，数据不出本地
- 🔍 **语义检索**：基于向量相似度，理解语义而非关键词匹配
- 💬 **智能问答**：结合知识库上下文 + LLM 生成准确回答
- 📂 **知识管理**：查看已入库文件，支持按来源删除
- 🔄 **增量更新**：新文件上传后自动入库，无需重建索引

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

编辑 `config.py`，或设置环境变量：

```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "sk-xxx"

# 或 Linux/Mac
export OPENAI_API_KEY="sk-xxx"
```

如果使用第三方兼容 API（如 DeepSeek）：

```bash
$env:OPENAI_BASE_URL = "https://api.deepseek.com"
$env:OPENAI_MODEL = "deepseek-chat"
```

### 3. 启动

```bash
streamlit run app.py
```

浏览器会自动打开 `http://localhost:8501`

## 项目结构

```
knowledge-base/
├── app.py              # Streamlit Web 主界面
├── config.py           # 配置文件（API Key、模型、参数）
├── parsers.py          # 文档解析器（TXT/PDF/Word/Excel）
├── vector_store.py     # 向量数据库操作（ChromaDB）
├── qa_chain.py         # 检索 + LLM 问答链
├── requirements.txt    # Python 依赖
├── uploads/            # 上传的原始文件
├── chroma_db/          # ChromaDB 向量数据库（自动生成）
├── data/               # 示例数据目录
└── README.md           # 本文件
```

## 技术栈

| 组件 | 选择 | 说明 |
|------|------|------|
| Embedding 模型 | bge-small-zh-v1.5 | 中文效果好，~90MB，CPU 可跑 |
| 向量数据库 | ChromaDB | 纯 Python，无需额外服务 |
| 文档解析 | PyMuPDF + python-docx + pandas | 覆盖主流格式 |
| LLM | OpenAI API | 兼容所有 OpenAI 格式 API |
| Web 界面 | Streamlit | 一行命令启动 |

## 使用提示

1. **首次使用**：上传几个文档后，系统会自动下载 Embedding 模型（约 90MB），之后离线可用
2. **文件命名**：建议用有意义的文件名，问答时会显示来源文件
3. **大文件**：系统会自动分块处理，单个文件大小建议不超过 50MB
4. **API 费用**：每次问答只消耗少量 token（检索到的片段 + 问题）

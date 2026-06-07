"""
向量数据库模块
负责：加载 Embedding 模型、管理 ChromaDB、入库、检索、文件管理
"""

import json
import logging
import os
import time
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Callable

import chromadb
import jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

import config

logger = logging.getLogger("kb")


class VectorStore:
    """向量知识库，封装了 Embedding 模型 + ChromaDB 的所有操作"""

    # 类级别模型单例，确保模型只加载一次
    _model = None
    _reranker = None
    _RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

    def __init__(self, progress_callback: Optional[Callable[[str], None]] = None):
        """
        初始化向量库（启动时预加载所有模型，之后操作即时响应）

        Args:
            progress_callback: 进度回调函数，接收状态消息字符串。
        """
        self._progress = progress_callback or (lambda msg: None)
        self._stats_cache = {"data": None, "timestamp": 0}
        self._stats_cache_ttl = 30

        # 预加载 Embedding 模型
        try:
            self._progress("🧠 加载 Embedding 模型...")
            self.model = self._get_model()
            self._progress("✅ Embedding 模型就绪")
        except Exception as e:
            raise RuntimeError(f"Embedding 模型加载失败: {self._diagnose_model_error(e)}") from e

        # 初始化 ChromaDB
        try:
            self.client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
            self.collection = self.client.get_or_create_collection(
                name=config.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            raise RuntimeError(f"ChromaDB 初始化失败: {self._diagnose_chroma_error(e)}") from e

        # 预加载 BM25 索引
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_docs: List[str] = []
        self._bm25_tokenized_docs: List[List[str]] = []
        self._progress("📚 构建 BM25 索引...")
        self._rebuild_bm25_index()

        # 文件注册表（JSON 持久化，管理文件元数据和分类）
        self._registry_path = os.path.join(config.CHROMA_PERSIST_DIR, "file_registry.json")
        self._registry: Dict[str, Dict] = self._load_registry()

        # 迁移：把 ChromaDB 中已有但注册表里没有的文件补录进来
        self._migrate_existing_files()

        self._progress("✅ 所有模型加载完成")

    # ==================== 文件注册表 ====================

    def _load_registry(self) -> Dict[str, Dict]:
        """从 JSON 文件加载文件注册表"""
        if os.path.exists(self._registry_path):
            try:
                with open(self._registry_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"加载文件注册表失败: {e}")
        return {}

    def _save_registry(self):
        """保存文件注册表到 JSON 文件"""
        try:
            with open(self._registry_path, "w", encoding="utf-8") as f:
                json.dump(self._registry, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存文件注册表失败: {e}")
            pass

    def register_file(self, source: str, chunk_count: int, file_size: int = 0, category: str = "未分类",
                      parent_chunks: List[str] = None):
        """注册一个新文件到注册表"""
        self._registry[source] = {
            "category": category,
            "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "chunk_count": chunk_count,
            "file_size": file_size,
        }
        if parent_chunks:
            self._registry[source]["parent_chunks"] = parent_chunks
        self._save_registry()

    def _get_parent_text(self, source: str, parent_index: int) -> str:
        """从注册表延迟查找父块文本（不存 ChromaDB metadata）"""
        entry = self._registry.get(source, {})
        parents = entry.get("parent_chunks", [])
        if parents and 0 <= parent_index < len(parents):
            return parents[parent_index]
        return ""

    def unregister_file(self, source: str):
        """从注册表中移除文件"""
        if source in self._registry:
            del self._registry[source]
            self._save_registry()

    def list_files(self, category: str = None, keyword: str = None) -> List[Dict]:
        """
        列出所有已注册文件，支持按分类和关键词筛选
        返回: [{"source": "...", "category": "...", "upload_time": "...", "chunk_count": N, "file_size": N}, ...]
        """
        files = []
        for source, meta in self._registry.items():
            # 分类筛选
            if category and category != "全部" and meta.get("category") != category:
                continue
            # 关键词筛选
            if keyword and keyword.lower() not in source.lower():
                continue
            files.append({"source": source, **meta})
        # 按上传时间倒序
        files.sort(key=lambda x: x.get("upload_time", ""), reverse=True)
        return files

    def get_categories(self) -> List[str]:
        """获取所有分类列表"""
        cats = set(meta.get("category", "未分类") for meta in self._registry.values())
        return ["全部"] + sorted(cats)

    def update_category(self, source: str, category: str):
        """更新文件分类"""
        if source in self._registry:
            self._registry[source]["category"] = category
            self._save_registry()

    def batch_delete(self, sources: List[str]) -> int:
        """批量删除多个文件，返回删除的片段总数。只在最后重建一次 BM25。"""
        total_deleted = 0
        for source in sources:
            results = self.collection.get(where={"source": source})
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                self.unregister_file(source)
                total_deleted += len(results["ids"])
        # 所有文件删完后，统一重建一次 BM25 索引
        if total_deleted > 0:
            self._rebuild_bm25_index()
            self._stats_cache["data"] = None
        return total_deleted

    def _migrate_existing_files(self):
        """把 ChromaDB 中已有但注册表里没有的文件补录进来"""
        total = self.collection.count()
        if total == 0:
            return

        all_meta = self.collection.get(include=["metadatas"])
        # 统计每个 source 的片段数
        source_counts: Dict[str, int] = {}
        for meta in all_meta["metadatas"]:
            src = meta.get("source", "")
            if src:
                source_counts[src] = source_counts.get(src, 0) + 1

        migrated = 0
        for source, count in source_counts.items():
            if source not in self._registry:
                self._registry[source] = {
                    "category": "未分类",
                    "upload_time": "未知",
                    "chunk_count": count,
                    "file_size": 0,
                }
                migrated += 1

        if migrated > 0:
            self._save_registry()

    def _rebuild_bm25_index(self):
        """从 ChromaDB 重建 BM25 索引（同时缓存 metadata，避免重复查询）"""
        total = self.collection.count()
        if total == 0:
            self._bm25 = None
            self._bm25_docs = []
            self._bm25_tokenized_docs = []
            self._bm25_sources = []
            return

        all_data = self.collection.get(include=["documents", "metadatas"])
        self._bm25_docs = list(all_data["documents"])
        self._bm25_sources = [m.get("source", "未知") for m in all_data["metadatas"]]
        self._bm25_tokenized_docs = [list(jieba.lcut(doc)) for doc in self._bm25_docs]
        self._bm25 = BM25Okapi(self._bm25_tokenized_docs)

    def _append_bm25(self, chunks: List[str], source: str):
        """增量追加 BM25 索引（不需要全量重建）"""
        new_tokenized = [list(jieba.lcut(doc)) for doc in chunks]
        self._bm25_docs.extend(chunks)
        self._bm25_tokenized_docs.extend(new_tokenized)
        self._bm25_sources.extend([source] * len(chunks))
        self._bm25 = BM25Okapi(self._bm25_tokenized_docs)

    @classmethod
    def _get_model(cls):
        """获取模型单例（类方法缓存，确保模型只加载一次）"""
        if cls._model is None:
            # 设置 HuggingFace 镜像源（国内网络环境）
            if config.HF_MIRROR_URL:
                os.environ["HF_ENDPOINT"] = config.HF_MIRROR_URL
            try:
                # 优先从本地缓存加载（离线可用，不联网）
                cls._model = SentenceTransformer(
                    config.EMBEDDING_MODEL_NAME,
                    local_files_only=True
                )
            except Exception:
                # 本地没有则联网下载
                cls._model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        return cls._model

    @classmethod
    def _get_reranker(cls) -> Optional[CrossEncoder]:
        """
        获取 Reranker 单例（CrossEncoder）。
        优先从本地缓存加载，加载失败则返回 None（graceful fallback）。
        """
        if cls._reranker is not None:
            return cls._reranker

        if config.HF_MIRROR_URL:
            os.environ["HF_ENDPOINT"] = config.HF_MIRROR_URL

        try:
            # 优先从本地缓存加载
            cls._reranker = CrossEncoder(
                cls._RERANKER_MODEL_NAME,
                local_files_only=True
            )
        except Exception:
            try:
                # 本地没有则联网下载
                cls._reranker = CrossEncoder(cls._RERANKER_MODEL_NAME)
            except Exception:
                # Reranker 加载失败，graceful fallback
                cls._reranker = None

        return cls._reranker

    def _rerank(self, query: str, hits: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        使用 CrossEncoder 对候选文档重排序，显著提升精度。

        Args:
            query: 用户查询
            hits: 候选文档列表 [{"text": ..., "source": ..., "score": ...}, ...]
            top_k: 重排序后返回的文档数量
        Returns:
            重排序后的文档列表
        """
        reranker = self._get_reranker()
        if reranker is None or not hits:
            # Reranker 不可用时，直接按原始分数截断返回
            return hits[:top_k]

        # 构建 (query, document) 配对列表
        pairs = [(query, hit["text"]) for hit in hits]
        # CrossEncoder 打分
        scores = reranker.predict(pairs)

        # 将 reranker 分数附加到 hits 上
        for hit, score in zip(hits, scores):
            hit["rerank_score"] = float(score)

        # 按 reranker 分数降序排序
        hits.sort(key=lambda x: x["rerank_score"], reverse=True)

        return hits[:top_k]

    def _diagnose_model_error(self, error: Exception) -> str:
        """诊断模型加载错误，提供中文详细诊断信息"""
        error_str = str(error).lower()

        if "connection" in error_str or "timeout" in error_str or "network" in error_str:
            return (
                "网络连接问题：无法下载模型文件。\n"
                "建议：\n"
                "  1. 检查网络连接\n"
                f"  2. 设置 HuggingFace 镜像源：export HF_ENDPOINT={config.HF_MIRROR_URL}\n"
                "     或在 .env 文件中设置：HF_MIRROR_URL=" + config.HF_MIRROR_URL + "\n"
                "  3. 或设置代理：export HTTP_PROXY=你的代理地址"
            )
        elif "disk" in error_str or "space" in error_str or "no space" in error_str:
            return (
                "磁盘空间不足：模型文件（约 90MB）无法保存。\n"
                "建议：\n"
                "  1. 清理磁盘空间\n"
                "  2. 模型缓存路径：" + (os.path.expanduser("~/.cache/huggingface") if hasattr(os, 'expanduser') else "~/.cache/huggingface")
            )
        elif "permission" in error_str or "access denied" in error_str:
            return (
                "权限不足：无法读取或写入模型文件。\n"
                "建议：\n"
                "  1. 检查文件权限\n"
                "  2. 以管理员身份运行"
            )
        elif "not found" in error_str or "404" in error_str:
            return (
                f"模型不存在：'{config.EMBEDDING_MODEL_NAME}' 在 HuggingFace 上未找到。\n"
                "建议：请检查 config.EMBEDDING_MODEL_NAME 配置是否正确"
            )
        else:
            return f"未知错误：{str(error)}"

    def _diagnose_chroma_error(self, error: Exception) -> str:
        """诊断 ChromaDB 错误，提供中文详细诊断信息"""
        error_str = str(error).lower()

        if "permission" in error_str or "access denied" in error_str:
            return (
                "权限不足：无法创建 ChromaDB 数据目录。\n"
                f"建议：检查目录权限或手动创建：{config.CHROMA_PERSIST_DIR}"
            )
        elif "disk" in error_str or "space" in error_str:
            return "磁盘空间不足，无法写入数据。请清理磁盘空间后重试。"
        else:
            return f"未知错误：{str(error)}"

    def _get_embedding(self, texts: List[str], batch_size: int = 256) -> List[List[float]]:
        """
        批量生成文本的向量表示（分批处理，避免内存溢出）
        """
        if not texts:
            return []

        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            self._progress(f"[向量库] 生成向量中... ({batch_num}/{total_batches})")
            embeddings = self.model.encode(batch, normalize_embeddings=True)
            all_embeddings.extend(embeddings.tolist())

        return all_embeddings

    @staticmethod
    def _make_id(text: str, source: str, extra: str = "") -> str:
        """根据文本内容和来源生成唯一 ID（用于去重）"""
        raw = f"{source}::{extra}::{text}" if extra else f"{source}::{text}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def add_documents(self, chunks: List[str], source: str, batch_size: int = 256,
                      file_size: int = 0, category: str = "未分类") -> int:
        """
        将一批文本片段入库（分批处理，避免内存溢出）。
        同时更新 BM25 索引。

        Args:
            chunks: 分块后的文本列表
            source: 来源文件名（用于溯源和去重）
            batch_size: 每批处理的片段数量，默认 256
        Returns:
            实际新增的片段数量
        """
        if not chunks:
            return 0

        total_added = 0
        total_batches = (len(chunks) + batch_size - 1) // batch_size

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_num = i // batch_size + 1
            self._progress(f"[向量库] 入库中... ({batch_num}/{total_batches})")

            # 生成向量
            embeddings = self._get_embedding(batch_chunks)

            # 生成 ID 和元数据
            start_idx = i
            ids = [self._make_id(chunk, source) for chunk in batch_chunks]
            metadatas = [{"source": source, "chunk_index": start_idx + j} for j in range(len(batch_chunks))]

            # 写入 ChromaDB（自动去重，相同 ID 会覆盖）
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=batch_chunks,
                metadatas=metadatas,
            )
            total_added += len(batch_chunks)

        # 增量更新 BM25 索引（不全量重建）
        self._append_bm25(chunks, source)

        # 注册文件到管理表
        self.register_file(source, total_added, file_size, category)

        # 清除统计缓存
        self._stats_cache["data"] = None

        return total_added

    def add_documents_hierarchical(self, parent_chunks: List[str], child_docs: List[Dict],
                                    source: str, batch_size: int = 256,
                                    file_size: int = 0, category: str = "未分类") -> int:
        """
        层级分块入库：将子块入库，同时记录其所属父块文本。

        Args:
            parent_chunks: 父块文本列表
            child_docs: 子块文档列表，每个元素为 dict，包含至少 "text" 和 "parent_index" 字段
            source: 来源文件名
            batch_size: 每批处理的片段数量，默认 256
            file_size: 文件大小（字节）
            category: 文件分类
        Returns:
            实际新增的子块数量
        """
        if not child_docs:
            return 0

        total_added = 0
        total_batches = (len(child_docs) + batch_size - 1) // batch_size

        for i in range(0, len(child_docs), batch_size):
            batch = child_docs[i:i + batch_size]
            batch_num = i // batch_size + 1
            self._progress(f"[向量库] 层级入库中... ({batch_num}/{total_batches})")

            # 提取子块文本，生成向量
            texts = [doc["text"] for doc in batch]
            embeddings = self._get_embedding(texts, batch_size=batch_size)

            # 生成 ID 和元数据（不存 parent_text，通过注册表延迟查找）
            ids = []
            metadatas = []
            for j, doc in enumerate(batch):
                parent_index = doc.get("parent_index", 0)
                doc_id = self._make_id(doc["text"], source, extra=f"{parent_index}_{i+j}")
                ids.append(doc_id)
                metadatas.append({
                    "source": source,
                    "chunk_index": i + j,
                    "parent_index": parent_index,
                    "chunk_type": "child",
                })

            # 写入 ChromaDB
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            total_added += len(batch)

        # 增量更新 BM25 索引
        child_texts = [doc["text"] for doc in child_docs]
        self._append_bm25(child_texts, source)

        # 注册文件到管理表（父块存注册表，不存 ChromaDB）
        self.register_file(source, total_added, file_size, category, parent_chunks=parent_chunks)

        # 清除统计缓存
        self._stats_cache["data"] = None

        return total_added

    def search_hyde(self, query: str, hypothetical_answer: str,
                    top_k: Optional[int] = None) -> List[Dict]:
        """
        HyDE 检索：同时用原始查询和假设性答案进行向量检索，用 RRF 融合结果。

        Args:
            query: 用户查询
            hypothetical_answer: LLM 生成的假设性答案
            top_k: 返回结果数量
        Returns:
            [{"text": "...", "child_text": "...", "source": "...", "score": ...}, ...]
        """
        k = top_k or config.TOP_K
        total_count = self.collection.count()
        if total_count == 0:
            return []

        candidate_k = min(20, total_count)

        # 分别为查询和假设性答案生成向量
        query_embedding = self._get_embedding([query])
        hyde_embedding = self._get_embedding([hypothetical_answer])

        # 用查询向量检索
        results_q = self.collection.query(
            query_embeddings=query_embedding,
            n_results=candidate_k,
            include=["documents", "metadatas", "distances"],
        )
        # 用假设性答案向量检索
        results_h = self.collection.query(
            query_embeddings=hyde_embedding,
            n_results=candidate_k,
            include=["documents", "metadatas", "distances"],
        )

        # RRF 融合
        RRF_K = 60
        all_docs: Dict[str, Dict] = {}

        for rank, idx in enumerate(range(len(results_q["ids"][0]))):
            meta = results_q["metadatas"][0][idx]
            source = meta.get("source", "未知")
            parent_index = meta.get("parent_index")
            parent_text = self._get_parent_text(source, parent_index) if parent_index is not None else ""
            doc_text = results_q["documents"][0][idx]
            display_text = parent_text if parent_text else doc_text
            child_text = doc_text if parent_text else None
            doc_key = display_text

            if doc_key not in all_docs:
                entry: Dict = {
                    "text": display_text,
                    "source": source,
                    "rrf_score": 0.0,
                }
                if child_text is not None:
                    entry["child_text"] = child_text
                all_docs[doc_key] = entry
            all_docs[doc_key]["rrf_score"] += 1.0 / (RRF_K + rank + 1)

        for rank, idx in enumerate(range(len(results_h["ids"][0]))):
            meta = results_h["metadatas"][0][idx]
            source = meta.get("source", "未知")
            parent_index = meta.get("parent_index")
            parent_text = self._get_parent_text(source, parent_index) if parent_index is not None else ""
            doc_text = results_h["documents"][0][idx]
            display_text = parent_text if parent_text else doc_text
            child_text = doc_text if parent_text else None
            doc_key = display_text

            if doc_key not in all_docs:
                entry = {
                    "text": display_text,
                    "source": meta.get("source", "未知"),
                    "rrf_score": 0.0,
                }
                if child_text is not None:
                    entry["child_text"] = child_text
                all_docs[doc_key] = entry
            all_docs[doc_key]["rrf_score"] += 1.0 / (RRF_K + rank + 1)

        # 按 RRF 分数排序，取 top candidate 数量
        sorted_docs = sorted(all_docs.values(), key=lambda x: x["rrf_score"], reverse=True)

        # Rerank 合并后的候选结果
        reranked = self._rerank(query, sorted_docs, top_k=k)

        # 构建最终结果
        result = []
        max_rrf = sorted_docs[0]["rrf_score"] if sorted_docs else 1.0
        for doc in reranked:
            normalized_score = doc["rrf_score"] / max_rrf if max_rrf > 0 else 0
            entry = {
                "text": doc["text"],
                "child_text": doc.get("child_text"),
                "source": doc["source"],
                "score": round(normalized_score, 4),
            }
            result.append(entry)

        return result

    def generate_document_summary(self, text: str, source: str, llm_client=None) -> str:
        """
        生成文档摘要。

        Args:
            text: 文档全文或前几块拼接的文本
            source: 文档来源名称
            llm_client: 可选的 LLM 客户端（需有 chat.completions.create 方法）
        Returns:
            2-3 句话的文档摘要
        """
        if llm_client is not None:
            try:
                response = llm_client.chat.completions.create(
                    model=config.OPENAI_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": "你是一个文档摘要助手。请用 2-3 句话概括以下文档的核心内容，语言简洁。",
                        },
                        {"role": "user", "content": f"文档来源: {source}\n\n{text[:3000]}"},
                    ],
                    temperature=0.3,
                    max_tokens=200,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"LLM 摘要生成失败，使用回退方案: {e}")

        # 回退：取前 500 字符
        return text[:500].strip()

    def search(self, query: str, top_k: Optional[int] = None, score_threshold: Optional[float] = None) -> List[Dict]:
        """
        语义检索：根据查询文本返回最相关的文档片段。
        先检索 top_k=20 个候选文档，再用 Reranker 重排序，返回最终的 top_k 个结果。
        返回: [{"text": "...", "source": "...", "score": 0.85}, ...]

        Args:
            query: 查询文本
            top_k: 返回的最相关文档片段数量
            score_threshold: 相似度分数阈值，低于此值的结果不返回
        """
        threshold = score_threshold if score_threshold is not None else getattr(config, "SCORE_THRESHOLD", 0.3)
        k = top_k or config.TOP_K
        total_count = self.collection.count()
        if total_count == 0:
            return []

        # 先检索更多候选文档（以便 Reranker 发挥效果）
        candidate_k = min(20, total_count)
        final_k = min(k, total_count)

        query_embedding = self._get_embedding([query])
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=candidate_k,
            include=["documents", "metadatas", "distances"]
        )

        hits = []
        for i in range(len(results["ids"][0])):
            # ChromaDB 返回的是 distance，cosine 距离越小越相似
            distance = results["distances"][0][i]
            score = 1 - distance  # 转换为相似度分数
            meta = results["metadatas"][0][i]
            source = meta.get("source", "未知")
            parent_index = meta.get("parent_index")
            parent_text = self._get_parent_text(source, parent_index) if parent_index is not None else ""
            if parent_text:
                hits.append({
                    "text": parent_text,
                    "child_text": results["documents"][0][i],
                    "source": source,
                    "score": round(score, 4),
                })
            else:
                hits.append({
                    "text": results["documents"][0][i],
                    "source": source,
                    "score": round(score, 4),
                })

        # Reranker 重排序后过滤并返回最终结果
        reranked = self._rerank(query, hits, top_k=final_k)
        if threshold is not None:
            reranked = [h for h in reranked if h.get("rerank_score", h["score"]) >= threshold]
        return reranked

    def search_hybrid(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        混合检索：同时执行向量检索和 BM25 检索，用 RRF 融合结果。
        """
        k = top_k or config.TOP_K
        total_count = self.collection.count()
        if total_count == 0:
            return []

        # 向量检索
        vector_hits = self.search(query, top_k=20, score_threshold=None)

        # BM25 检索
        bm25_hits = self._search_bm25(query, top_k=20)

        # 用 doc 文本作为 key 构建融合索引
        all_docs: Dict[str, Dict] = {}
        RRF_K = 60

        for rank, hit in enumerate(vector_hits):
            doc_key = hit["text"]
            if doc_key not in all_docs:
                all_docs[doc_key] = {"text": hit["text"], "source": hit["source"], "rrf_score": 0.0}
            all_docs[doc_key]["rrf_score"] += 1.0 / (RRF_K + rank + 1)
            all_docs[doc_key]["vector_score"] = hit["score"]
            all_docs[doc_key]["vector_rank"] = rank + 1

        for rank, hit in enumerate(bm25_hits):
            doc_key = hit["text"]
            if doc_key not in all_docs:
                all_docs[doc_key] = {"text": hit["text"], "source": hit["source"], "rrf_score": 0.0}
            all_docs[doc_key]["rrf_score"] += 1.0 / (RRF_K + rank + 1)
            all_docs[doc_key]["bm25_score"] = hit["bm25_score"]
            all_docs[doc_key]["bm25_rank"] = rank + 1

        # 按 RRF 分数降序排序
        sorted_docs = sorted(all_docs.values(), key=lambda x: x["rrf_score"], reverse=True)

        # 归一化 RRF 分数到 0~1 范围（方便和阈值比较）
        max_rrf = sorted_docs[0]["rrf_score"] if sorted_docs else 1.0

        # 构建最终结果
        result = []
        for doc in sorted_docs[:k]:
            normalized_score = doc["rrf_score"] / max_rrf if max_rrf > 0 else 0
            entry = {
                "text": doc["text"],
                "source": doc["source"],
                "score": round(normalized_score, 4),
                "vector_score": round(doc.get("vector_score", 0.0), 4),
                "bm25_score": round(doc.get("bm25_score", 0.0), 4),
            }
            result.append(entry)

        return result

    def _search_bm25(self, query: str, top_k: int = 20) -> List[Dict]:
        """
        BM25 检索

        Args:
            query: 查询文本
            top_k: 返回的最相关文档数量
        Returns:
            [{"text": "...", "source": "...", "bm25_score": ...}, ...]
        """
        if self._bm25 is None or not self._bm25_docs:
            return []

        # 对查询做 jieba 分词
        query_tokens = list(jieba.lcut(query))
        # 获取 BM25 分数
        scores = self._bm25.get_scores(query_tokens)
        # 获取 top_k 索引
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        hits = []
        for idx in top_indices:
            hits.append({
                "text": self._bm25_docs[idx],
                "source": self._bm25_sources[idx] if idx < len(self._bm25_sources) else "未知",
                "bm25_score": round(scores[idx], 4),
            })

        return hits

    def delete_by_source(self, source: str) -> int:
        """删除指定来源的所有文档片段"""
        results = self.collection.get(where={"source": source})
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            # 重建 BM25 索引
            self._rebuild_bm25_index()
            # 从注册表移除
            self.unregister_file(source)
            # 清除统计缓存
            self._stats_cache["data"] = None
        return len(results["ids"])

    def get_stats(self, use_cache: bool = True) -> Dict:
        """
        获取知识库统计信息（带缓存，避免频繁全量查询）

        Args:
            use_cache: 是否使用缓存，默认 True。缓存 30 秒内有效。
        Returns:
            包含 total_chunks、total_sources、sources 的字典
        """
        current_time = time.time()

        # 检查缓存是否有效
        if use_cache and self._stats_cache["data"] is not None:
            if current_time - self._stats_cache["timestamp"] < self._stats_cache_ttl:
                return self._stats_cache["data"]

        total = self.collection.count()
        # 获取所有来源（去重）
        if total > 0:
            all_meta = self.collection.get(include=["metadatas"])
            sources = set(m.get("source", "") for m in all_meta["metadatas"])
        else:
            sources = set()

        result = {
            "total_chunks": total,
            "total_sources": len(sources),
            "sources": sorted(sources),
        }

        # 更新缓存
        self._stats_cache["data"] = result
        self._stats_cache["timestamp"] = current_time

        return result
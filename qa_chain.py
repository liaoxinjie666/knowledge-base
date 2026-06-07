"""
问答链模块
负责：检索相关文档 → 拼装 Prompt → 调用 LLM 生成回答
"""

import logging
from typing import List, Dict, Optional
from openai import OpenAI, AuthenticationError

import config


# 配置日志
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


class QAChain:
    """基于检索增强生成（RAG）的问答链"""

    def _get_client_and_model(self):
        """
        延迟获取 OpenAI 客户端和模型配置
        每次调用时读取最新配置，响应运行时配置变更
        优先从 streamlit session_state 读取（界面上修改的配置），
        回退到 config 模块默认值（环境变量）
        """
        try:
            import streamlit as st
            api_key = st.session_state.get("api_key") or config.OPENAI_API_KEY
            base_url = st.session_state.get("base_url") or config.OPENAI_BASE_URL
            model = st.session_state.get("model") or config.OPENAI_MODEL
        except Exception:
            # 非 Streamlit 环境（如单元测试），直接读 config
            api_key = config.OPENAI_API_KEY
            base_url = config.OPENAI_BASE_URL
            model = config.OPENAI_MODEL

        if not api_key:
            raise ValueError("API Key 为空，请在左侧面板填写正确的 API Key")

        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs), model

    def _build_prompt(self, question: str, context_docs: List[Dict]):
        """构造 Prompt 和来源列表"""
        context_parts = []
        sources = []
        for i, doc in enumerate(context_docs, 1):
            context_parts.append(f"[{i}] (来源: {doc['source']})\n{doc['text']}")
            sources.append(doc["source"])

        context_text = "\n\n".join(context_parts)
        system_prompt = """你是一个知识库问答助手。请根据提供的参考文档和对话历史回答用户的问题。

规则：
1. 优先根据提供的参考文档回答，不要编造信息
2. 如果参考文档中没有相关信息，结合对话历史和你的知识回答
3. 注意理解上下文：当用户说"它""这个""那个"等指代词时，根据对话历史判断指的是什么
4. 回答要准确、简洁、有条理
5. 使用 [1][2] 等标注来指示引用的参考文档编号
6. 在回答末尾列出参考来源列表"""
        user_prompt = f"参考文档：\n{context_text}\n\n用户问题：{question}\n\n请根据以上参考文档回答问题。"
        return system_prompt, user_prompt, sources

    def generate_hypothetical_answer(self, question: str) -> str:
        """
        生成假设性回答（HyDE），用于提升检索匹配效果
        - question: 用户的问题
        - 返回: 假设性回答文本，失败时返回原始问题
        """
        try:
            client, model = self._get_client_and_model()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个知识库助手。"},
                    {"role": "user", "content": f"请为以下问题生成一个简短的假设性回答（用于搜索匹配，不需要完全准确）：\n{question}"},
                ],
                temperature=0.7,
                max_tokens=300,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"generate_hypothetical_answer failed: {e}")
            return question

    def generate_multi_queries(self, question: str, count: int = 3) -> List[str]:
        """
        将问题改写为多个搜索查询变体
        - question: 原始问题
        - count: 生成的查询数量
        - 返回: 查询变体列表，失败时返回包含原始问题的列表
        """
        try:
            client, model = self._get_client_and_model()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个搜索查询改写助手。"},
                    {"role": "user", "content": f"请将以下问题改写为 {count} 个不同的搜索查询变体，每行一个，不要编号：\n{question}"},
                ],
                temperature=0.8,
                max_tokens=200,
            )
            content = response.choices[0].message.content.strip()
            queries = [line.strip() for line in content.split("\n") if line.strip()]
            return queries[:count]
        except Exception as e:
            logger.error(f"generate_multi_queries failed: {e}")
            return [question]

    def assess_relevance(self, question: str, docs: List[Dict]) -> List[Dict]:
        """
        评估文档与问题的相关性，过滤不相关文档
        - question: 用户的问题
        - docs: 检索到的文档列表
        - 返回: 过滤后的相关文档列表
        """
        try:
            client, model = self._get_client_and_model()
            relevant_docs = []
            for doc in docs:
                snippet = doc["text"][:300]
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "你是一个文档相关性评估助手。"},
                        {"role": "user", "content": f"问题：{question}\n\n文档片段：{snippet}\n\n该文档片段与问题是否相关？请只回答「相关」或「不相关」。"},
                    ],
                    temperature=0.0,
                    max_tokens=10,
                )
                result = response.choices[0].message.content.strip().lower()
                if "相关" in result and "不相关" not in result:
                    relevant_docs.append(doc)
            if not relevant_docs:
                return [docs[0]]
            return relevant_docs
        except Exception as e:
            logger.error(f"assess_relevance failed: {e}")
            return docs

    def extract_conversation_memory(self, chat_history: List[Dict]) -> str:
        """
        从对话历史中提取关键信息作为记忆关键词
        - chat_history: 对话历史列表
        - 返回: 逗号分隔的关键词字符串（最多50字符），失败时返回空字符串
        """
        try:
            recent = chat_history[-6:]
            history_text = "\n".join(
                f"{msg['role']}: {msg['content']}" for msg in recent
            )
            client, model = self._get_client_and_model()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个对话分析助手。"},
                    {"role": "user", "content": f"请从以下对话历史中提取关键实体、话题和用户偏好，以逗号分隔的关键词形式返回（不超过50个字符）：\n{history_text}"},
                ],
                temperature=0.3,
                max_tokens=50,
            )
            return response.choices[0].message.content.strip()[:50]
        except Exception as e:
            logger.error(f"extract_conversation_memory failed: {e}")
            return ""

    def answer_stream(self, question: str, context_docs: List[Dict],
                      chat_history: Optional[List[Dict]] = None):
        """
        流式生成回答（generator，逐 token 返回）
        - question: 用户的问题
        - context_docs: 向量检索返回的相关片段列表
        - chat_history: 多轮对话历史，格式 [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]
        - yield: 每次返回一个 token 字符串
        - 最后 yield {"sources": [...]} 表示结束
        """
        system_prompt, user_prompt, sources = self._build_prompt(question, context_docs)

        # 构造完整的消息列表：system + 历史对话 + 当前用户问题
        messages = [{"role": "system", "content": system_prompt}]
        if chat_history:
            # 只保留最近 5 轮对话（10 条消息），避免 token 超限
            recent_history = chat_history[-10:]
            messages.extend(recent_history)
        messages.append({"role": "user", "content": user_prompt})

        try:
            client, model = self._get_client_and_model()
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=2000,
                stream=True,  # 开启流式输出
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
            # 最后返回来源信息
            yield {"sources": list(set(sources))}
        except ValueError as e:
            yield str(e)
        except AuthenticationError as e:
            logger.error(f"LLM authentication failed: {e}")
            yield "API Key 无效或为空，请在左侧面板填写正确的 API Key"
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            yield "抱歉，AI 服务暂时不可用，请稍后重试"

    def answer_simple_stream(self, question: str):
        """
        不使用知识库，流式回答（兜底方案）
        """
        try:
            client, model = self._get_client_and_model()
            stream = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个有帮助的助手。"},
                    {"role": "user", "content": question},
                ],
                temperature=0.7,
                max_tokens=2000,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except ValueError as e:
            yield str(e)
        except AuthenticationError as e:
            logger.error(f"LLM authentication failed: {e}")
            yield "API Key 无效或为空，请在左侧面板填写正确的 API Key"
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            yield "抱歉，AI 服务暂时不可用，请稍后重试"

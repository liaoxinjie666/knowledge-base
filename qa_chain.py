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
            context_parts.append(f"[片段{i}] (来源: {doc['source']})\n{doc['text']}")
            sources.append(doc["source"])

        context_text = "\n\n".join(context_parts)
        system_prompt = """你是一个知识库问答助手。请根据提供的参考文档和对话历史回答用户的问题。

规则：
1. 优先根据提供的参考文档回答，不要编造信息
2. 如果参考文档中没有相关信息，结合对话历史和你的知识回答
3. 注意理解上下文：当用户说"它""这个""那个"等指代词时，根据对话历史判断指的是什么
4. 回答要准确、简洁、有条理
5. 适当引用来源"""
        user_prompt = f"参考文档：\n{context_text}\n\n用户问题：{question}\n\n请根据以上参考文档回答问题。"
        return system_prompt, user_prompt, sources

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

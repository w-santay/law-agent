"""
知识检索Agent — 法律知识库RAG问答
负责从向量数据库中检索相关法律法规、司法解释、判例等文档，结合上下文生成准确法律回答。
实现完整的RAG流程：Query改写 → 向量检索 → 重排序 → 上下文注入 → 生成回答。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from memory.long_term import LongTermMemory
from tracing.otel_config import trace_agent_call


RAG_SYSTEM_PROMPT = """你是一个专业的法律知识库问答Agent，负责根据检索到的法律法规、司法解释等文档回答用户法律问题。

回答规则：
1. 严格基于检索到的法律法规和司法解释内容回答，不要编造法律条文
2. 如果检索结果中没有相关法律依据，明确告知用户并建议咨询专业律师
3. 回答要准确专业，引用具体的法律条文和司法解释
4. 对于法律意见，必须标注“以上内容仅供参考，不构成正式法律意见，建议咨询专业律师”
5. 在回答末尾标注引用的法律法规来源

回答格式：
- 先引用相关法律条文或司法解释
- 对条文进行专业解读
- 结合用户具体情况给出适用分析
- 添加免责声明和风险提示
"""

QUERY_REWRITE_PROMPT = """请将用户的口语化法律问题改写为更适合法律知识库检索的查询语句。
保留核心法律语义，去除口语化表达，补充法律术语和法规名称。
只返回改写后的查询，不要其他内容。

用户原始问题: {query}
"""


class KnowledgeRAGAgent:
    """法律知识检索Agent - 实现完整RAG流程"""

    def __init__(self, llm: ChatOpenAI, long_term_memory: LongTermMemory | None = None):
        self.llm = llm
        self.long_term_memory = long_term_memory or LongTermMemory()

    @trace_agent_call("rag_query_rewrite")
    async def rewrite_query(self, original_query: str) -> str:
        """Query改写：将口语化法律问题转为检索友好的法规查询"""
        messages = [
            HumanMessage(content=QUERY_REWRITE_PROMPT.format(query=original_query)),
        ]
        response = await self.llm.ainvoke(messages)
        return response.content.strip()

    @trace_agent_call("rag_retrieve")
    async def retrieve_documents(self, query: str, top_k: int = 5) -> list[dict]:
        """从向量数据库检索相关法律法规文档"""
        docs = self.long_term_memory.search(query, top_k=top_k)
        return docs

    @trace_agent_call("rag_rerank")
    async def rerank_documents(
        self, query: str, documents: list[dict], top_k: int = 3
    ) -> list[dict]:
        """对检索结果重排序，提升相关性"""
        if not documents:
            return []

        doc_summaries = "\n".join(
            f"[{i}] {doc.get('content', '')[:200]}"
            for i, doc in enumerate(documents)
        )

        messages = [
            SystemMessage(content="你是一个法律文档相关性排序专家。"),
            HumanMessage(content=(
                f"用户查询: {query}\n\n"
                f"候选文档:\n{doc_summaries}\n\n"
                f"请返回最相关的{top_k}个文档的索引号，用逗号分隔，如: 0,2,4"
            )),
        ]

        response = await self.llm.ainvoke(messages)

        try:
            indices = [int(i.strip()) for i in response.content.split(",")]
            reranked = [documents[i] for i in indices if i < len(documents)]
        except (ValueError, IndexError):
            reranked = documents[:top_k]

        return reranked

    @trace_agent_call("rag_generate")
    async def generate_answer(self, query: str, context_docs: list[dict]) -> str:
        """基于检索文档生成回答"""
        if not context_docs:
            return "抱歉，法律知识库中暂未找到与您问题相关的法律法规。建议您咨询专业律师获取更准确的法律意见。"

        context = "\n\n---\n\n".join(
            f"来源: {doc.get('source', '未知')}\n内容: {doc.get('content', '')}"
            for doc in context_docs
        )

        messages = [
            SystemMessage(content=RAG_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"用户问题: {query}\n\n"
                f"检索到的参考文档:\n{context}"
            )),
        ]

        response = await self.llm.ainvoke(messages)
        return response.content

    @trace_agent_call("knowledge_rag_process")
    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        完整RAG流程（作为Graph节点）：
        1. Query改写
        2. 向量检索
        3. 重排序
        4. 生成回答
        """
        messages = state.get("messages", [])
        if not messages:
            return state

        original_query = messages[-1].content

        rewritten_query = await self.rewrite_query(original_query)

        raw_docs = await self.retrieve_documents(rewritten_query, top_k=5)

        reranked_docs = await self.rerank_documents(rewritten_query, raw_docs, top_k=3)

        answer = await self.generate_answer(original_query, reranked_docs)

        return {
            **state,
            "sub_results": {
                **state.get("sub_results", {}),
                "knowledge_rag": answer,
            },
        }

"""
意图路由Agent — 法律场景用户意图识别与分类
负责分析用户输入，识别出具体的法律意图，为Supervisor提供路由依据。
支持多级意图分类：一级意图(法律咨询/合同审查/诉讼辅助/法规检索/合规审查) -> 二级意图(具体法律领域)。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from tracing.otel_config import trace_agent_call


class IntentCategory(str, Enum):
    """一级意图分类"""
    LEGAL_CONSULTATION = "legal_consultation"   # 法律咨询类
    CONTRACT_REVIEW = "contract_review"         # 合同审查类
    LITIGATION_SUPPORT = "litigation_support"   # 诉讼辅助类
    REGULATION_SEARCH = "regulation_search"     # 法规检索类
    COMPLIANCE_AUDIT = "compliance_audit"       # 合规审查类
    UNKNOWN = "unknown"


@dataclass
class IntentResult:
    """意图识别结果"""
    primary_intent: IntentCategory
    secondary_intent: str
    confidence: float
    entities: dict[str, str]
    suggested_agent: str


INTENT_SYSTEM_PROMPT = """你是一个专业的法律意图识别Agent，负责分析用户的法律咨询消息。

请从以下维度分析用户意图：
1. 一级意图分类: legal_consultation(法律咨询), contract_review(合同审查), litigation_support(诉讼辅助), regulation_search(法规检索), compliance_audit(合规审查)
2. 二级意图: 具体的法律领域子类型（如：劳动法、合同纠纷、知识产权、公司法等）
3. 置信度: 0.0-1.0
4. 关键实体: 提取合同编号、法律法规名称、案件类型、当事人等关键信息
5. 建议路由: knowledge_rag(法律知识检索), case_handler(案件/合同处理), compliance_checker(合规审查)

以JSON格式返回，示例：
{
    "primary_intent": "legal_consultation",
    "secondary_intent": "labor_law",
    "confidence": 0.95,
    "entities": {"law_name": "劳动合同法", "issue": "加班工资争议"},
    "suggested_agent": "knowledge_rag"
}

法律场景特殊规则：
- 涉及合同审查、合同纠纷、合同起草 → case_handler
- 涉及法规检索、法律条文解读 → knowledge_rag
- 涉及合规风险、法规遵循、审计 → compliance_checker
- 涉及诉讼、仲裁、证据分析 → case_handler
"""


class IntentRouterAgent:
    """意图路由Agent"""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    @trace_agent_call("intent_router")
    async def classify(self, user_message: str) -> IntentResult:
        """对用户消息进行意图分类"""
        messages = [
            SystemMessage(content=INTENT_SYSTEM_PROMPT),
            HumanMessage(content=f"用户消息: {user_message}"),
        ]

        response = await self.llm.ainvoke(messages)

        import json
        try:
            result = json.loads(response.content)
        except json.JSONDecodeError:
            result = {
                "primary_intent": "unknown",
                "secondary_intent": "unknown",
                "confidence": 0.0,
                "entities": {},
                "suggested_agent": "knowledge_rag",
            }


        suggested = result.get("suggested_agent", "knowledge_rag")
        valid_agents = {"knowledge_rag", "case_handler", "compliance_checker"}
        if suggested not in valid_agents:
            suggested = "knowledge_rag"

        return IntentResult(
            primary_intent=IntentCategory(result.get("primary_intent", "unknown")),
            secondary_intent=result.get("secondary_intent", "unknown"),
            confidence=result.get("confidence", 0.0),
            entities=result.get("entities", {}),
            suggested_agent=suggested,
        )

    @trace_agent_call("intent_router_process")
    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """作为Graph节点处理状态"""
        messages = state.get("messages", [])
        if not messages:
            return state

        last_message = messages[-1].content if messages else ""
        intent_result = await self.classify(last_message)

        return {
            **state,
            "intent": intent_result.suggested_agent,
            "sub_results": {
                **state.get("sub_results", {}),
                "intent_router": {
                    "primary": intent_result.primary_intent.value,
                    "secondary": intent_result.secondary_intent,
                    "confidence": intent_result.confidence,
                    "entities": intent_result.entities,
                },
            },
        }

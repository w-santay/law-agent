"""
案件/合同处理Agent — 法律案件管理与合同审查
负责创建法律案件、查询案件进度、审查合同条款、辅助诉讼准备。
通过MCP工具协议调用外部法律系统。
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from tracing.otel_config import trace_agent_call


class CaseStatus(str, Enum):
    CREATED = "created"           # 已创建
    REVIEWING = "reviewing"       # 审查中
    PENDING_REVIEW = "pending_review"  # 待审核
    IN_PROGRESS = "in_progress"   # 进行中
    RESOLVED = "resolved"         # 已解决
    CLOSED = "closed"             # 已关闭
    ESCALATED = "escalated"       # 已升级至律师


class CasePriority(str, Enum):
    LOW = "low"           # 一般咨询
    MEDIUM = "medium"     # 常规案件
    HIGH = "high"         # 重要案件
    URGENT = "urgent"     # 紧急案件（如诉讼时效即将届满）


class CaseType(str, Enum):
    CONTRACT_REVIEW = "contract_review"      # 合同审查
    LITIGATION_SUPPORT = "litigation_support"  # 诉讼辅助
    LEGAL_OPINION = "legal_opinion"          # 法律意见书
    COMPLIANCE_CHECK = "compliance_check"    # 合规审查
    DISPUTE_RESOLUTION = "dispute_resolution"  # 争议解决
    GENERAL = "general"                      # 通用


CASE_SYSTEM_PROMPT = """你是一个专业的法律案件处理Agent，负责处理用户的法律案件和合同审查请求。

你的职责：
1. 分析用户需求，判断是否需要创建法律案件或审查合同
2. 提取案件关键信息（案件类型、优先级、法律领域、描述）
3. 创建法律案件并返回案件编号
4. 查询现有案件进度

案件类型：
- contract_review: 合同审查（合同条款风险分析、合同起草辅助）
- litigation_support: 诉讼辅助（证据整理、诉讼策略、法律文书）
- legal_opinion: 法律意见书（特定法律问题的专业分析）
- compliance_check: 合规审查（企业合规风险检查）
- dispute_resolution: 争议解决（调解、仲裁方案建议）
- general: 通用法律咨询

优先级判断规则：
- urgent: 诉讼时效即将届满、紧急财产保全、涉嫌犯罪
- high: 重要合同签署、重大诉讼案件、知识产权侵权
- medium: 常规法律咨询、一般合同审查
- low: 法律信息咨询、法规解读

请以JSON格式返回案件信息：
{
    "action": "create|query|review",
    "case_type": "contract_review|litigation_support|legal_opinion|...",
    "priority": "low|medium|high|urgent",
    "legal_area": "合同法|劳动法|知识产权|公司法|...",
    "summary": "案件摘要",
    "details": "详细描述",
    "key_issues": ["关键法律问题列表"]
}
"""


class CaseStore:
    """内存案件存储（生产环境应替换为数据库）"""

    def __init__(self):
        self._cases: dict[str, dict] = {}

    def create(
        self,
        case_type: str,
        priority: str,
        legal_area: str,
        summary: str,
        details: str,
        key_issues: list[str],
        user_id: str,
    ) -> dict:
        case_id = f"CASE-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        case = {
            "case_id": case_id,
            "type": case_type,
            "priority": priority,
            "legal_area": legal_area,
            "status": CaseStatus.CREATED.value,
            "summary": summary,
            "details": details,
            "key_issues": key_issues,
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        self._cases[case_id] = case
        return case

    def query(self, case_id: str) -> dict | None:
        return self._cases.get(case_id)

    def query_by_user(self, user_id: str) -> list[dict]:
        return [c for c in self._cases.values() if c["user_id"] == user_id]

    def update_status(self, case_id: str, status: str) -> dict | None:
        case = self._cases.get(case_id)
        if case:
            case["status"] = status
            case["updated_at"] = datetime.now().isoformat()
        return case


class CaseHandlerAgent:
    """法律案件/合同处理Agent"""

    def __init__(self, llm: ChatOpenAI, case_store: CaseStore | None = None):
        self.llm = llm
        self.case_store = case_store or CaseStore()

    @trace_agent_call("case_analyze")
    async def analyze_request(self, user_message: str) -> dict:
        """分析用户需求，提取案件信息"""
        messages = [
            SystemMessage(content=CASE_SYSTEM_PROMPT),
            HumanMessage(content=f"用户消息: {user_message}"),
        ]

        response = await self.llm.ainvoke(messages)

        import json
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {
                "action": "create",
                "case_type": "general",
                "priority": "medium",
                "legal_area": "综合",
                "summary": user_message[:100],
                "details": user_message,
                "key_issues": [],
            }

    @trace_agent_call("case_create")
    async def create_case(self, case_info: dict, user_id: str) -> str:
        """创建法律案件"""
        case = self.case_store.create(
            case_type=case_info.get("case_type", "general"),
            priority=case_info.get("priority", "medium"),
            legal_area=case_info.get("legal_area", "综合"),
            summary=case_info.get("summary", ""),
            details=case_info.get("details", ""),
            key_issues=case_info.get("key_issues", []),
            user_id=user_id,
        )

        priority_label = {
            "low": "一般", "medium": "中等", "high": "重要", "urgent": "紧急"
        }.get(case["priority"], "中等")

        key_issues_text = "\n".join(f"  - {issue}" for issue in case.get("key_issues", []))

        return (
            f"法律案件已创建成功！\n\n"
            f"📋 案件编号: {case['case_id']}\n"
            f"📝 案件类型: {case['type']}\n"
            f"⚖️ 法律领域: {case['legal_area']}\n"
            f"⚡ 优先级: {priority_label}\n"
            f"📄 案件摘要: {case['summary']}\n"
            f"🔍 关键法律问题:\n{key_issues_text}\n"
            f"🕐 创建时间: {case['created_at']}\n\n"
            f"我们将尽快安排专业律师处理您的案件，请保存好案件编号以便后续查询。"
        )

    @trace_agent_call("case_query")
    async def query_case(self, case_id: str) -> str:
        """查询案件状态"""
        case = self.case_store.query(case_id)
        if not case:
            return f"未找到案件编号 {case_id}，请确认案件编号是否正确。"

        status_label = {
            "created": "已创建",
            "reviewing": "审查中",
            "pending_review": "待审核",
            "in_progress": "进行中",
            "resolved": "已解决",
            "closed": "已关闭",
            "escalated": "已升级至律师",
        }.get(case["status"], case["status"])

        return (
            f"案件查询结果：\n\n"
            f"📋 案件编号: {case['case_id']}\n"
            f"📊 状态: {status_label}\n"
            f"📝 类型: {case['type']}\n"
            f"⚖️ 法律领域: {case['legal_area']}\n"
            f"📄 案件摘要: {case['summary']}\n"
            f"🕐 创建时间: {case['created_at']}\n"
            f"🔄 更新时间: {case['updated_at']}"
        )

    @trace_agent_call("case_handler_process")
    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """作为Graph节点处理状态"""
        messages = state.get("messages", [])
        user_id = state.get("user_id", "anonymous")

        if not messages:
            return state

        last_message = messages[-1].content
        case_info = await self.analyze_request(last_message)

        action = case_info.get("action", "create")

        if action == "query" and "case_id" in case_info:
            result = await self.query_case(case_info["case_id"])
        else:
            result = await self.create_case(case_info, user_id)

        return {
            **state,
            "sub_results": {
                **state.get("sub_results", {}),
                "case_handler": result,
            },
        }
"""
MCP工具协议服务端 — 法律Agent系统Model Context Protocol实现
遵循Anthropic MCP标准，通过JSON-RPC 2.0提供工具注册/发现/调用能力。
支持动态工具扩展，法律Agent通过统一接口调用外部法律系统。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable
from datetime import datetime


@dataclass
class ToolDefinition:
    """MCP工具定义"""
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Awaitable[Any]]
    category: str = "general"
    requires_auth: bool = False


@dataclass
class ToolCallResult:
    """工具调用结果"""
    tool_name: str
    success: bool
    result: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class MCPToolServer:
    """
    MCP工具服务端。

    实现 Model Context Protocol 的核心功能：
    1. 工具注册 (Tool Registration)
    2. 工具发现 (Tool Discovery) - Agent可查询可用工具列表
    3. 工具调用 (Tool Invocation) - 通过JSON-RPC 2.0协议调用
    4. 结果返回 (Result Delivery)

    遵循MCP规范：
    - 使用JSON-RPC 2.0消息格式
    - 支持工具的inputSchema声明
    - 提供标准化的错误码
    """

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        self._call_log: list[ToolCallResult] = []

    def register_tool(self, tool: ToolDefinition) -> None:
        """注册一个MCP工具"""
        self._tools[tool.name] = tool

    def register(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        category: str = "general",
        requires_auth: bool = False,
    ) -> Callable:
        """工具注册装饰器"""
        def decorator(func: Callable[..., Awaitable[Any]]) -> Callable:
            tool = ToolDefinition(
                name=name,
                description=description,
                input_schema=input_schema,
                handler=func,
                category=category,
                requires_auth=requires_auth,
            )
            self._tools[name] = tool
            return func
        return decorator

    def list_tools(self, category: str | None = None) -> list[dict]:
        """
        工具发现：列出所有可用工具。
        对应MCP的 tools/list 方法。
        """
        tools = []
        for tool in self._tools.values():
            if category and tool.category != category:
                continue
            tools.append({
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema,
                "category": tool.category,
            })
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        """
        工具调用：执行指定工具。
        对应MCP的 tools/call 方法。
        """
        import time

        tool = self._tools.get(name)
        if tool is None:
            result = ToolCallResult(
                tool_name=name,
                success=False,
                error=f"Tool '{name}' not found. Available: {list(self._tools.keys())}",
            )
            self._call_log.append(result)
            return result

        start = time.time()
        try:
            output = await tool.handler(**arguments)
            duration_ms = (time.time() - start) * 1000

            result = ToolCallResult(
                tool_name=name,
                success=True,
                result=output,
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            result = ToolCallResult(
                tool_name=name,
                success=False,
                error=str(e),
                duration_ms=duration_ms,
            )

        self._call_log.append(result)
        return result

    async def handle_jsonrpc(self, request: dict) -> dict:
        """
        处理JSON-RPC 2.0请求。
        MCP协议传输层实现。
        """
        method = request.get("method", "")
        params = request.get("params", {})
        req_id = request.get("id", 1)

        try:
            if method == "tools/list":
                result = self.list_tools(category=params.get("category"))
            elif method == "tools/call":
                tool_name = params.get("name", "")
                arguments = params.get("arguments", {})
                call_result = await self.call_tool(tool_name, arguments)
                result = {
                    "success": call_result.success,
                    "result": call_result.result,
                    "error": call_result.error,
                }
            elif method == "ping":
                result = {"status": "ok"}
            else:
                return {
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                    "id": req_id,
                }

            return {"jsonrpc": "2.0", "result": result, "id": req_id}

        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": str(e)},
                "id": req_id,
            }

    def get_call_log(self, last_n: int = 100) -> list[dict]:
        """获取最近的工具调用日志"""
        return [
            {
                "tool": r.tool_name,
                "success": r.success,
                "duration_ms": r.duration_ms,
                "timestamp": r.timestamp,
                "error": r.error,
            }
            for r in self._call_log[-last_n:]
        ]


def create_default_tools(server: MCPToolServer) -> MCPToolServer:
    """注册默认的法律MCP工具集"""

    @server.register(
        name="regulation_search",
        description="搜索法律法规库，返回相关法律条文和司法解释",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "法律法规检索查询（如：劳动合同解除、合同违约责任）"},
                "law_name": {"type": "string", "description": "指定法律名称（如：民法典、劳动合同法、刑法）"},
                "top_k": {"type": "integer", "description": "返回数量", "default": 3},
            },
            "required": ["query"],
        },
        category="legal_knowledge",
    )
    async def regulation_search(query: str, law_name: str = "", top_k: int = 3) -> list[dict]:
        return [
            {"content": f"关于'{query}'的{law_name or '法律法规'}条文", "source": "法律数据库", "score": 0.95},
        ]

    @server.register(
        name="case_search",
        description="检索相似判例，返回相关案例信息",
        input_schema={
            "type": "object",
            "properties": {
                "case_type": {"type": "string", "description": "案件类型（如：劳动争议、合同纠纷、知识产权）"},
                "keyword": {"type": "string", "description": "关键搜索词"},
                "court_level": {"type": "string", "description": "法院级别（如：最高法、省高院、基层法院）"},
            },
            "required": ["case_type"],
        },
        category="legal_case",
    )
    async def case_search(case_type: str, keyword: str = "", court_level: str = "") -> list[dict]:
        return [
            {
                "case_id": "CASE-2025-001",
                "title": f"{case_type}典型案例",
                "court": court_level or "最高人民法院",
                "judgment": "判例摘要内容",
                "relevance": 0.90,
            },
        ]

    @server.register(
        name="contract_review",
        description="合同条款智能审查，检测风险条款和法律合规性",
        input_schema={
            "type": "object",
            "properties": {
                "contract_text": {"type": "string", "description": "合同文本内容"},
                "contract_type": {"type": "string", "description": "合同类型（如：劳动合同、买卖合同、租赁合同）"},
                "review_focus": {"type": "string", "description": "审查重点（如：违约责任、霸王条款、知识产权）"},
            },
            "required": ["contract_text", "contract_type"],
        },
        category="contract",
    )
    async def contract_review(contract_text: str, contract_type: str, review_focus: str = "") -> dict:
        return {
            "risk_level": "medium",
            "issues_found": ["违约责任条款不够明确", "争议解决条款缺失"],
            "suggestions": ["建议补充违约责任的具体量化标准", "建议明确争议解决方式和管辖法院"],
            "compliance_status": "需要修改",
        }

    @server.register(
        name="statute_limitation_check",
        description="诉讼时效计算工具 — 根据案件类型和事发日期计算时效",
        input_schema={
            "type": "object",
            "properties": {
                "case_type": {"type": "string", "description": "案件类型"},
                "incident_date": {"type": "string", "description": "事发日期（YYYY-MM-DD格式）"},
                "law_reference": {"type": "string", "description": "适用的法律法规"},
            },
            "required": ["case_type", "incident_date"],
        },
        category="legal_calculator",
    )
    async def statute_limitation_check(case_type: str, incident_date: str, law_reference: str = "") -> dict:
        return {
            "case_type": case_type,
            "standard_period": "3年",
            "start_date": incident_date,
            "expiry_date": "需根据具体日期计算",
            "remaining_days": "需根据当前日期计算",
            "law_basis": law_reference or "《民法典》第188条",
            "warning": "诉讼时效即将届满请尽快行动",
        }

    @server.register(
        name="legal_risk_assessment",
        description="法律风险评估 — 评估特定行为或决策的法律风险等级",
        input_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "description": "待评估的行为或决策"},
                "industry": {"type": "string", "description": "行业领域"},
                "jurisdiction": {"type": "string", "description": "管辖区域（如：中国大陆、香港、美国）"},
            },
            "required": ["action"],
        },
        category="compliance",
    )
    async def legal_risk_assessment(action: str, industry: str = "", jurisdiction: str = "中国大陆") -> dict:
        risk_level = "low"
        if any(kw in action for kw in ["侵权", "违约", "欺诈", "违规"]):
            risk_level = "high"
        elif any(kw in action for kw in ["合同", "投资", "转让"]):
            risk_level = "medium"
        return {
            "action": action,
            "risk_level": risk_level,
            "key_regulations": ["相关法律法规列表"],
            "compliance_requirements": ["合规要求列表"],
            "requires_legal_counsel": risk_level in ("high", "critical"),
        }

    return server

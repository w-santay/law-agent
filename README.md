# 多Agent智能法律助手 (Law-Agent)

基于 LangGraph 的 Supervisor 编排多 Agent 法律智能系统，提供法律咨询、合同审查、案件管理、法规检索、合规审查等核心能力。

## 架构概览

```
用户法律咨询 → Supervisor(路由决策) → 法律子Agent处理 → 合规审查 → 汇总回复
                    │                       │                │
                    ├── intent_router       │                │
                    ├── knowledge_rag ──────┤────────────────┤
                    ├── case_handler ───────┤                │
                    └── compliance_checker ─┘ ← 所有回复必经 │
```

系统采用 LangGraph StateGraph 实现，由 Supervisor 作为中央协调者，根据法律咨询意图将请求路由到对应子 Agent，所有子 Agent 的输出都必须经过合规审查后才能返回给用户。

## 核心模块

| 模块 | 说明 |
|------|------|
| `agents/supervisor.py` | Supervisor 编排 Agent — 法律咨询路由决策与结果汇总 |
| `agents/intent_router.py` | 意图路由 Agent — 法律意图分类（法律咨询/合同审查/诉讼辅助/法规检索/合规审查） |
| `agents/knowledge_rag.py` | 法律知识检索 Agent — 完整 RAG 流程（Query改写→法规检索→重排→生成） |
| `agents/case_handler.py` | 案件/合同处理 Agent — 法律案件创建、查询、合同审查 |
| `agents/compliance_checker.py` | 合规审查 Agent — 两阶段审查（规则引擎 + LLM 深度审查） |
| `memory/working_memory.py` | 工作记忆 — 进程内存，维护当前对话中间推理状态 |
| `memory/short_term.py` | 短期记忆 — Redis 会话缓存，保留最近 N 轮对话 |
| `memory/long_term.py` | 长期记忆 — FAISS 向量检索，支持法律法规语义搜索 |
| `mcp/mcp_server.py` | MCP 工具协议 — 法律工具注册/发现/调用，遵循 JSON-RPC 2.0 |
| `tracing/otel_config.py` | 全链路追踪 — OpenTelemetry 集成，支持 Jaeger 可视化 |
| `config/settings.py` | 应用配置 — Pydantic Settings，通过环境变量或 .env 加载 |
| `api/main.py` | FastAPI 入口 — REST API + 对话历史 + MCP 法律工具调用 + 指标查询 |

## 技术栈

- **LLM 框架**: LangGraph + LangChain + LangChain-OpenAI
- **API 服务**: FastAPI + Uvicorn
- **向量数据库**: FAISS (可切换 Milvus)
- **会话缓存**: Redis
- **可观测性**: OpenTelemetry + Jaeger
- **工具协议**: MCP (Model Context Protocol)
- **配置管理**: Pydantic Settings + python-dotenv
- **容器化**: Docker + Docker Compose

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/w-santay/law-agent.git
cd law-agent

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env，填入必要配置
# 必须配置: OPENAI_API_KEY
```

关键配置项：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `OPENAI_API_KEY` | OpenAI API Key（必填） | - |
| `OPENAI_BASE_URL` | API Base URL | `https://api.openai.com/v1` |
| `MODEL_NAME` | 模型名称 | `gpt-4o` |
| `REDIS_URL` | Redis 连接地址 | `redis://localhost:6379/0` |
| `FAISS_INDEX_PATH` | FAISS 索引路径 | `./vector_store/faiss_index` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP 收集器地址 | `http://localhost:4317` |
| `HOST` | 服务监听地址 | `0.0.0.0` |
| `PORT` | 服务监听端口 | `8000` |

### 3. 启动依赖服务

```bash
# 启动 Jaeger（全链路追踪可视化）
docker compose up jaeger

# 如需 Redis（短期记忆），取消 docker-compose.yml 中 redis 的注释后启动
```

### 4. 启动服务

```bash
# 直接运行
python -m api.main

# 或使用 uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Docker 部署

```bash
# 构建并启动全部服务（取消 docker-compose.yml 中相关注释）
docker compose up --build
```

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/chat` | POST | 主法律咨询接口 |
| `/api/history/{session_id}` | GET | 获取对话历史 |
| `/api/tools` | GET | MCP 法律工具发现 |
| `/api/tools/call` | POST | MCP 法律工具调用 |
| `/api/metrics` | GET | 获取系统指标 |
| `/health` | GET | 健康检查 |

### 法律咨询接口示例

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "劳动合同解除有哪些法律规定？", "user_id": "user_001"}'
```

响应：

```json
{
  "response": "根据《劳动合同法》第36-37条...",
  "session_id": "uuid-xxx",
  "intent": "knowledge_rag",
  "compliance_passed": true
}
```

### MCP 法律工具列表

| 工具 | 类别 | 说明 |
|------|------|------|
| `regulation_search` | 法律知识 | 搜索法律法规库，返回条文和司法解释 |
| `case_search` | 案件检索 | 检索相似判例，返回相关案例信息 |
| `contract_review` | 合同审查 | 合同条款智能审查，检测风险条款 |
| `statute_limitation_check` | 时效计算 | 诉讼时效计算工具 |
| `legal_risk_assessment` | 合规评估 | 法律风险评估工具 |

## 项目结构

```
law-agent/
├── agents/              # Agent 实现
│   ├── supervisor.py    # Supervisor 编排
│   ├── intent_router.py # 法律意图路由
│   ├── knowledge_rag.py # 法律知识检索 RAG
│   ├── case_handler.py  # 案件/合同处理
│   └── compliance_checker.py # 法律合规审查
├── api/                 # FastAPI 入口
│   └── main.py          # REST API + SSE
├── config/              # 配置管理
│   └── settings.py      # Pydantic Settings
├── memory/              # 记忆系统
│   ├── working_memory.py# 工作记忆（进程内存）
│   ├── short_term.py    # 短期记忆（Redis）
│   └── long_term.py     # 长期记忆（FAISS向量）
├── mcp/                 # MCP 工具协议
│   └── mcp_server.py    # 法律工具注册/发现/调用
├── tracing/             # 全链路追踪
│   └── otel_config.py   # OpenTelemetry 配置
├── .env.example         # 环境变量模板
├── Dockerfile           # Docker 构建
├── docker-compose.yml   # Docker Compose 配置
└── requirements.txt     # Python 依赖
```

## 可观测性

系统通过 OpenTelemetry 实现全链路追踪，每个 Agent 调用都会创建 Span 记录：

- **agent.name**: Agent 名称
- **agent.duration_ms**: 调用耗时
- **agent.success**: 是否成功
- **agent.error**: 错误信息（如失败）

启动 Jaeger 后访问 `http://localhost:16686` 查看追踪数据。

## 注意事项

- `OPENAI_API_KEY` 为必填配置，否则 LLM 调用将失败
- Redis 为可选依赖，未配置时短期记忆将回退到进程内存
- FAISS 向量索引在首次运行时自动创建，生产环境建议使用 OpenAI Embedding API 替换简易嵌入
- 合规审查为强制环节，所有 Agent 输出必须经过合规检查才能返回给用户
- 案件存储当前为内存实现，生产环境应替换为数据库
- **重要免责声明**: 本系统提供的法律信息仅供参考，不构成正式法律意见。涉及具体法律问题建议咨询专业律师。

## License

MIT
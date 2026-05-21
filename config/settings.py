"""
应用配置 — 通过环境变量或 .env 文件加载
"""

from pydantic_settings import BaseSettings

# 所有的环境变量都有对应的字段定义，Pydantic 就能正确验证和加载配置了
class Settings(BaseSettings):
    # LLM
    api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    model_name: str = "gpt-4o"


    # Redis (短期记忆)
    redis_url: str = "redis://localhost:6379/0"

    # Vector Store (长期记忆)
    vector_store_type: str = "faiss"
    faiss_index_path: str = "./vector_store/faiss_index"
    milvus_host: str = "localhost"
    milvus_port: int = 19530

    # OpenTelemetry (全链路追踪)
    otel_service_name: str = "law-multi-agent"
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"
    otel_traces_sampler: str = "always_on"

    # LangSmith (可选)
    langsmith_api_key: str = ""
    langsmith_project: str = "law-agent"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
if __name__ == '__main__':
    print(settings.api_key)

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class AppConfig:
    """Application configuration using environment variables"""
    
    # MCP Gateway configuration
    mcp_gateway_endpoint: str = os.getenv("MCPGATEWAY_ENDPOINT", "http://localhost:8811/sse")
    
    # Model configuration - automatically injected by Docker Compose
    recommendation_model_url: str = os.getenv("RECOMMENDATION_MODEL_URL", "http://localhost:8000/v1")
    recommendation_model_name: str = os.getenv("RECOMMENDATION_MODEL_NAME", "ai/qwen2.5:3B-Q4_0")
    
    analysis_model_url: str = os.getenv("ANALYSIS_MODEL_URL", "http://localhost:8000/v1")
    analysis_model_name: str = os.getenv("ANALYSIS_MODEL_NAME", "ai/qwen2.5:3B-Q4_0")
    
    # Provider configuration
    model_provider: str = os.getenv("MODEL_PROVIDER", "dmr")
    
    # GitHub configuration
    github_token_file: Optional[str] = os.getenv("GITHUB_TOKEN_FILE")
    
    # Application settings
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    app_title: str = os.getenv("APP_TITLE", "Project Recommender Agents")
    
    @property
    def github_token(self) -> Optional[str]:
        """Read GitHub token from file if available"""
        if self.github_token_file and os.path.exists(self.github_token_file):
            with open(self.github_token_file, 'r') as f:
                return f.read().strip()
        return os.getenv("GITHUB_TOKEN")

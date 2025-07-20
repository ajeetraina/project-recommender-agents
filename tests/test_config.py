import os
from src.config import AppConfig

def test_app_config():
    """Test application configuration"""
    config = AppConfig()
    
    assert config.app_title == "Project Recommender Agents"
    assert config.log_level == "INFO"
    assert "localhost" in config.mcp_gateway_endpoint

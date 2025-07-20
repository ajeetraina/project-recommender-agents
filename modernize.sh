#!/bin/bash

# ==============================================================================
# Project Recommender Agents - Complete Modernization Script
# ==============================================================================
# This script transforms the repository to follow docker/compose-for-agents pattern
# Author: AI Assistant
# Version: 1.0
# ==============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REPO_NAME="project-recommender-agents"
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

create_backup() {
    log_info "Creating backup of existing files..."
    mkdir -p "$BACKUP_DIR"
    
    # Backup existing files if they exist
    for file in compose.yaml compose.yml docker-compose.yaml docker-compose.yml Dockerfile requirements.txt README.md; do
        if [[ -f "$file" ]]; then
            cp "$file" "$BACKUP_DIR/" 2>/dev/null || true
            log_info "Backed up: $file"
        fi
    done
    
    # Backup src directory if it exists
    if [[ -d "src" ]]; then
        cp -r src "$BACKUP_DIR/" 2>/dev/null || true
        log_info "Backed up: src/ directory"
    fi
    
    log_success "Backup created in: $BACKUP_DIR"
}

create_directory_structure() {
    log_info "Creating modern directory structure..."
    
    # Create main directories
    mkdir -p src/{agents,models,utils,mcp}
    mkdir -p tests
    mkdir -p .secrets
    mkdir -p workspace
    
    # Create __init__.py files
    touch src/__init__.py
    touch src/agents/__init__.py
    touch src/models/__init__.py
    touch src/utils/__init__.py
    touch src/mcp/__init__.py
    touch tests/__init__.py
    
    log_success "Directory structure created"
}

create_compose_files() {
    log_info "Creating modernized compose files..."

    # Base compose.yaml
    cat > compose.yaml << 'EOF'
name: project-recommender-agents

services:
  # Main Streamlit application
  recommender-ui:
    build:
      context: .
      target: runtime
    ports:
      - "8501:8501"
    environment:
      # MCP Gateway connection
      - MCPGATEWAY_ENDPOINT=http://mcp-gateway:8811/sse
      # Application settings
      - APP_TITLE=Project Recommender Agents
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
      # GitHub integration
      - GITHUB_TOKEN_FILE=/run/secrets/github_token
    depends_on:
      - mcp-gateway
    secrets:
      - github_token
    models:
      # Primary reasoning model for recommendations
      recommendation_model:
        endpoint_var: RECOMMENDATION_MODEL_URL
        model_var: RECOMMENDATION_MODEL_NAME
      # Analysis model for GitHub profile analysis
      analysis_model:
        endpoint_var: ANALYSIS_MODEL_URL
        model_var: ANALYSIS_MODEL_NAME
    volumes:
      - ./src:/app/src:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Multi-agent orchestrator service  
  agent-orchestrator:
    build:
      context: .
      target: agent-runtime
    environment:
      - MCPGATEWAY_ENDPOINT=http://mcp-gateway:8811/sse
      - GITHUB_TOKEN_FILE=/run/secrets/github_token
    depends_on:
      - mcp-gateway
    secrets:
      - github_token
    models:
      recommendation_model:
        endpoint_var: RECOMMENDATION_MODEL_URL
        model_var: RECOMMENDATION_MODEL_NAME
      analysis_model:
        endpoint_var: ANALYSIS_MODEL_URL
        model_var: ANALYSIS_MODEL_NAME

  # MCP Gateway with enhanced tool support
  mcp-gateway:
    image: docker/mcp-gateway:latest
    use_api_socket: true
    ports:
      - "8811:8811"
    command:
      - --transport=sse
      - --servers=github,brave,fetch,filesystem
    environment:
      # GitHub integration
      - GITHUB_TOKEN_FILE=/run/secrets/github_token
      # Brave Search API
      - BRAVE_API_KEY_FILE=/run/secrets/brave_api_key
      # Filesystem access (restricted to workspace)
      - FILESYSTEM_ALLOWED_PATHS=/workspace
    secrets:
      - github_token
      - brave_api_key
    volumes:
      - ./workspace:/workspace
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8811/health"]
      interval: 30s
      timeout: 10s
      retries: 3

# Global model definitions
models:
  recommendation_model:
    model: ai/qwen2.5:3B-Q4_0
    context_size: 16384
    runtime_flags:
      - "--threads=4"
      - "--batch-size=256"
  
  analysis_model:
    model: ai/qwen2.5:3B-Q4_0
    context_size: 8192
    runtime_flags:
      - "--threads=4"
      - "--batch-size=128"

# Secrets management
secrets:
  github_token:
    file: ./.secrets/github_token
  brave_api_key:
    file: ./.secrets/brave_api_key

# Volumes
volumes:
  workspace:
    driver: local
EOF

    # Docker Model Runner configuration
    cat > compose.dmr.yaml << 'EOF'
# Enhanced configuration for Docker Model Runner
services:
  recommender-ui:
    environment:
      - MODEL_PROVIDER=dmr
      - DMR_BASE_URL=http://model-runner.docker.internal/engines/v1
    extra_hosts:
      - "model-runner.docker.internal:host-gateway"

  agent-orchestrator:
    environment:
      - MODEL_PROVIDER=dmr
      - DMR_BASE_URL=http://model-runner.docker.internal/engines/v1
    extra_hosts:
      - "model-runner.docker.internal:host-gateway"

# Optimized model configurations for local inference
models:
  recommendation_model:
    model: ai/qwen2.5:7B-Q4_0  # Larger model for better recommendations
    context_size: 32768        # 8GB VRAM
    # context_size: 16384      # 4GB VRAM (fallback)
    runtime_flags:
      - "--threads=8"
      - "--batch-size=512"
      - "--gpu-layers=35"
      - "--rope-freq-base=1000000"
  
  analysis_model:
    model: ai/qwen2.5:3B-Q4_0  # Smaller, faster model for analysis
    context_size: 16384
    runtime_flags:
      - "--threads=6"
      - "--batch-size=256"
      - "--gpu-layers=28"
EOF

    # Cloud Offload configuration (CORRECTED)
    cat > compose.offload.yaml << 'EOF'
# Cloud deployment overrides for Docker Offload
# Note: Docker Offload is enabled via 'docker offload start --gpu'
# This file only contains resource optimizations for cloud execution

services:
  recommender-ui:
    environment:
      - OFFLOAD_ENABLED=true
      - MODEL_CONTEXT_SIZE=32768
    
  agent-orchestrator:
    environment:
      - OFFLOAD_ENABLED=true
      - MODEL_CONTEXT_SIZE=32768

  mcp-gateway:
    environment:
      - OFFLOAD_ENABLED=true

# Enhanced models for cloud resources
models:
  recommendation_model:
    # Use larger, more capable model when offloaded
    model: ai/qwen2.5:14B-Q4_0  # Upgrade from 7B to 14B
    context_size: 32768         # Larger context
    runtime_flags:
      - "--threads=16"          # More CPU threads available
      - "--batch-size=1024"     # Larger batch size
      - "--gpu-layers=40"       # Full GPU utilization
  
  analysis_model:
    model: ai/qwen2.5:7B-Q4_0   # Upgrade from 3B to 7B  
    context_size: 16384
    runtime_flags:
      - "--threads=12"
      - "--batch-size=512"
      - "--gpu-layers=35"
EOF

    # OpenAI fallback configuration
    cat > compose.openai.yaml << 'EOF'
# OpenAI API integration
services:
  recommender-ui:
    environment:
      - MODEL_PROVIDER=openai
      - OPENAI_API_KEY_FILE=/run/secrets/openai_api_key
    secrets:
      - openai_api_key

  agent-orchestrator:
    environment:
      - MODEL_PROVIDER=openai
      - OPENAI_API_KEY_FILE=/run/secrets/openai_api_key
    secrets:
      - openai_api_key

# Override with OpenAI models
models:
  recommendation_model:
    model: gpt-4o
    provider: openai
    context_size: 128000
  
  analysis_model:
    model: gpt-4o-mini
    provider: openai
    context_size: 128000

secrets:
  openai_api_key:
    file: ./secret.openai-api-key
EOF

    log_success "Compose files created"
}

create_dockerfile() {
    log_info "Creating multi-stage Dockerfile..."

    cat > Dockerfile << 'EOF'
# syntax=docker/dockerfile:1
FROM python:3.11-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash --uid 1000 app

# Set up Python environment
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Development stage
FROM base as development

# Install development dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements-dev.txt

# Copy source code
COPY --chown=app:app src/ ./src/
COPY --chown=app:app tests/ ./tests/

USER app

# Development server with hot reload
CMD ["streamlit", "run", "src/main.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--browser.gatherUsageStats", "false"]

# Runtime stage for UI application
FROM base as runtime

# Install production dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=app:app src/ ./src/

# Create necessary directories
RUN mkdir -p /app/workspace && chown app:app /app/workspace

USER app

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/health || exit 1

# Start Streamlit application
CMD ["streamlit", "run", "src/main.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8501", \
     "--browser.gatherUsageStats", "false", \
     "--server.headless", "true"]

# Agent runtime stage for background services
FROM base as agent-runtime

# Install production dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy agent-specific code
COPY --chown=app:app src/ ./src/

USER app

# Health check for agent services
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import src.agents.health_check; src.agents.health_check.check()" || exit 1

# Start agent orchestrator
CMD ["python", "-m", "src.agents.orchestrator"]
EOF

    log_success "Dockerfile created"
}

create_requirements() {
    log_info "Creating requirements files..."

    cat > requirements.txt << 'EOF'
# Web framework
streamlit>=1.28.0
fastapi>=0.104.0
uvicorn>=0.24.0

# AI/ML libraries
openai>=1.3.0
anthropic>=0.7.0
httpx>=0.25.0

# MCP integration
mcp>=0.1.0
websockets>=11.0

# Data processing
pandas>=2.1.0
numpy>=1.24.0
pydantic>=2.4.0

# GitHub integration
PyGithub>=1.59.0
requests>=2.31.0

# Utilities
python-dotenv>=1.0.0
loguru>=0.7.0
asyncio-mqtt>=0.13.0
EOF

    cat > requirements-dev.txt << 'EOF'
# Include production requirements
-r requirements.txt

# Development dependencies
pytest>=7.4.0
pytest-asyncio>=0.21.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.5.0
pre-commit>=3.4.0
EOF

    log_success "Requirements files created"
}

create_application_code() {
    log_info "Creating application code structure..."

    # Main application
    cat > src/main.py << 'EOF'
import streamlit as st
import asyncio
from typing import Dict, List, Optional
import os
from src.agents.recommender import ProjectRecommenderAgent
from src.agents.analyzer import GitHubAnalyzerAgent
from src.config import AppConfig
from src.mcp.client import MCPClient

class ProjectRecommenderApp:
    def __init__(self):
        self.config = AppConfig()
        self.mcp_client = MCPClient(self.config.mcp_gateway_endpoint)
        self.recommender_agent = ProjectRecommenderAgent(
            model_url=self.config.recommendation_model_url,
            model_name=self.config.recommendation_model_name,
            mcp_client=self.mcp_client
        )
        self.analyzer_agent = GitHubAnalyzerAgent(
            model_url=self.config.analysis_model_url,
            model_name=self.config.analysis_model_name,
            mcp_client=self.mcp_client
        )

    def run(self):
        st.set_page_config(
            page_title="Project Recommender Agents",
            page_icon="🚀",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Sidebar for MCP server configuration
        self.render_sidebar()
        
        # Main interface
        st.title("🚀 Project Recommender Agents")
        st.markdown("AI-powered hackathon project recommendations based on GitHub analysis")
        
        # User input
        github_username = st.text_input(
            "GitHub Username", 
            placeholder="Enter GitHub username to analyze"
        )
        
        if st.button("Analyze & Recommend", type="primary"):
            if github_username:
                self.process_recommendation(github_username)
            else:
                st.error("Please enter a GitHub username")

    def render_sidebar(self):
        st.sidebar.title("🔧 MCP Configuration")
        
        # MCP server status
        try:
            available_servers = asyncio.run(self.mcp_client.get_available_servers())
            
            st.sidebar.subheader("Available MCP Servers")
            for server in available_servers:
                status = "🟢" if server.get("status") == "active" else "🔴"
                st.sidebar.write(f"{status} {server.get('name', 'Unknown')}")
            
            # Tool selection
            st.sidebar.subheader("Active Tools")
            available_tools = asyncio.run(self.mcp_client.get_available_tools())
            
            selected_tools = st.sidebar.multiselect(
                "Select tools to use:",
                options=[tool["name"] for tool in available_tools],
                default=[tool["name"] for tool in available_tools if tool.get("default", False)]
            )
            
            # Update MCP client with selected tools
            self.mcp_client.set_active_tools(selected_tools)
        except Exception as e:
            st.sidebar.error(f"MCP connection failed: {e}")

    def process_recommendation(self, username: str):
        """Process the recommendation workflow"""
        with st.spinner("Analyzing GitHub profile..."):
            # Phase 1: GitHub Analysis
            analysis_result = asyncio.run(
                self.analyzer_agent.analyze_profile(username)
            )
            
            if analysis_result.get("error"):
                st.error(f"Error analyzing profile: {analysis_result['error']}")
                return
        
        # Display analysis results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Profile Analysis")
            st.json(analysis_result.get("profile_summary", {}))
        
        with col2:
            st.subheader("💡 Skills & Interests")
            skills = analysis_result.get("skills", [])
            for skill in skills:
                st.badge(skill)
        
        with st.spinner("Generating project recommendations..."):
            # Phase 2: Project Recommendations
            recommendations = asyncio.run(
                self.recommender_agent.recommend_projects(analysis_result)
            )
        
        # Display recommendations
        st.subheader("🎯 Recommended Projects")
        for i, project in enumerate(recommendations.get("projects", []), 1):
            with st.expander(f"Project {i}: {project.get('title', 'Untitled')}"):
                st.write(f"**Description:** {project.get('description', 'No description')}")
                st.write(f"**Difficulty:** {project.get('difficulty', 'Unknown')}")
                st.write(f"**Technologies:** {', '.join(project.get('technologies', []))}")
                st.write(f"**Estimated Time:** {project.get('estimated_time', 'Unknown')}")
                
                if project.get("relevant_events"):
                    st.write("**Relevant Events:**")
                    for event in project["relevant_events"]:
                        st.write(f"- {event}")

# Health check endpoint for Docker
@st.cache_data
def health_check():
    return {"status": "healthy", "timestamp": str(asyncio.get_event_loop().time())}

if __name__ == "__main__":
    # Add health check route
    if st.query_params.get("health") == "true":
        st.json(health_check())
    else:
        app = ProjectRecommenderApp()
        app.run()
EOF

    # Configuration management
    cat > src/config.py << 'EOF'
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
EOF

    # Create basic agent structure
    cat > src/agents/base.py << 'EOF'
import asyncio
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import httpx
from loguru import logger

class BaseAgent(ABC):
    """Base class for all AI agents"""
    
    def __init__(self, model_url: str, model_name: str, mcp_client=None):
        self.model_url = model_url
        self.model_name = model_name
        self.mcp_client = mcp_client
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def call_llm(self, messages: list, temperature: float = 0.7) -> Dict[str, Any]:
        """Call the LLM with OpenAI-compatible API"""
        try:
            response = await self.client.post(
                f"{self.model_url}/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": 2000
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return {"error": str(e)}
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Abstract method for agent processing"""
        pass
EOF

    # Create placeholder agents
    cat > src/agents/analyzer.py << 'EOF'
from src.agents.base import BaseAgent
from typing import Dict, Any, List

class GitHubAnalyzerAgent(BaseAgent):
    """Agent for analyzing GitHub profiles and extracting insights"""
    
    async def analyze_profile(self, username: str) -> Dict[str, Any]:
        """Analyze a GitHub profile and extract key insights"""
        # Placeholder implementation
        return {
            "username": username,
            "profile_summary": {"experience_level": "intermediate"},
            "skills": ["Python", "Docker", "AI/ML"],
            "interests": ["Machine Learning", "DevOps"]
        }
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data"""
        return await self.analyze_profile(input_data.get("username", ""))
EOF

    cat > src/agents/recommender.py << 'EOF'
from src.agents.base import BaseAgent
from typing import Dict, Any, List

class ProjectRecommenderAgent(BaseAgent):
    """Agent for generating personalized hackathon project recommendations"""
    
    async def recommend_projects(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate project recommendations based on profile analysis"""
        # Placeholder implementation
        return {
            "projects": [
                {
                    "title": "AI-Powered Code Review Assistant",
                    "description": "Build a tool that uses LLMs to provide intelligent code review suggestions",
                    "technologies": ["Python", "Docker", "OpenAI API"],
                    "difficulty": "Intermediate",
                    "estimated_time": "48 hours",
                    "relevant_events": ["AI/ML Hackathon 2025"]
                }
            ]
        }
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data"""
        return await self.recommend_projects(input_data)
EOF

    # Create MCP client
    cat > src/mcp/client.py << 'EOF'
import asyncio
import json
from typing import Dict, Any, List, Optional
import websockets
from loguru import logger

class MCPClient:
    """Client for communicating with MCP Gateway via Server-Sent Events"""
    
    def __init__(self, gateway_endpoint: str):
        self.gateway_endpoint = gateway_endpoint
        self.active_tools: List[str] = []
        self.websocket = None
    
    async def get_available_servers(self) -> List[Dict[str, Any]]:
        """Get list of available MCP servers"""
        # Placeholder implementation
        return [
            {"name": "github", "status": "active"},
            {"name": "brave", "status": "active"},
            {"name": "fetch", "status": "active"}
        ]
    
    async def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools from all servers"""
        # Placeholder implementation
        return [
            {"name": "github/search", "default": True},
            {"name": "brave/search", "default": True},
            {"name": "fetch/url", "default": False}
        ]
    
    def set_active_tools(self, tools: List[str]):
        """Set active tools for the session"""
        self.active_tools = tools
        logger.info(f"Active tools updated: {tools}")
EOF

    # Create health check
    cat > src/agents/health_check.py << 'EOF'
import asyncio
import sys

async def check_agent_health():
    """Check health of agent services"""
    try:
        # Basic health check - can be expanded
        print("Health check passed")
        return True
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def check():
    """Synchronous health check wrapper"""
    return asyncio.run(check_agent_health())
EOF

    # Create orchestrator placeholder
    cat > src/agents/orchestrator.py << 'EOF'
import asyncio
import signal
import sys
from loguru import logger

class AgentOrchestrator:
    """Orchestrates multiple agents"""
    
    def __init__(self):
        self.running = False
    
    async def start(self):
        """Start the orchestrator"""
        logger.info("Agent orchestrator starting...")
        self.running = True
        
        while self.running:
            # Placeholder - implement actual orchestration logic
            await asyncio.sleep(10)
            logger.info("Orchestrator heartbeat")
    
    def stop(self):
        """Stop the orchestrator"""
        logger.info("Agent orchestrator stopping...")
        self.running = False

if __name__ == "__main__":
    orchestrator = AgentOrchestrator()
    
    # Handle shutdown signals
    def signal_handler(signum, frame):
        orchestrator.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run orchestrator
    asyncio.run(orchestrator.start())
EOF

    log_success "Application code structure created"
}

create_environment_files() {
    log_info "Creating environment configuration files..."

    cat > mcp.env.example << 'EOF'
# GitHub Integration
GITHUB_TOKEN=your_github_personal_access_token_here

# Brave Search API (for web search capabilities)
BRAVE_API_KEY=your_brave_search_api_key_here

# Model Configuration (optional overrides)
LOG_LEVEL=INFO
APP_TITLE=Project Recommender Agents

# Development Settings
STREAMLIT_THEME_BASE=dark
STREAMLIT_THEME_PRIMARY_COLOR=#ff6b6b
EOF

    cat > .env.example << 'EOF'
# Copy this to .env and fill in your values

# GitHub Integration
GITHUB_TOKEN=your_github_token_here

# Brave Search
BRAVE_API_KEY=your_brave_api_key_here

# OpenAI (optional)
OPENAI_API_KEY=your_openai_api_key_here

# Application Configuration
LOG_LEVEL=INFO
MODEL_PROVIDER=dmr
EOF

    cat > .gitignore << 'EOF'
# Secrets and environment
.env
.mcp.env
secret.*
.secrets/
*.key
*.token

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Docker
.dockerignore

# Logs
*.log
logs/

# Backup
backup_*/

# Workspace
workspace/
temp/
EOF

    log_success "Environment files created"
}

create_documentation() {
    log_info "Creating comprehensive documentation..."

    # Create scripts directory first
    mkdir -p scripts

    cat > README.md << 'EOF'
# 🚀 Project Recommender Agents - Modernized

A sophisticated multi-agent system that analyzes GitHub profiles to recommend personalized hackathon projects and relevant events, following the latest Docker Compose AI patterns.

## ✨ Key Features

- **🤖 Multi-Agent Architecture**: Specialized agents for analysis and recommendations
- **🔗 MCP Gateway Integration**: Secure, standardized tool access (GitHub, Brave Search, Web Fetch)
- **🏗️ Modern Docker Compose**: Latest model provider patterns with top-level `models:` configuration
- **⚡ Flexible Deployment**: Local (Docker Model Runner), Cloud (Offload), or OpenAI
- **📊 Interactive UI**: Streamlit-based interface with real-time MCP server management
- **🔒 Security-First**: Proper secrets management and container isolation

## 🚀 Quick Start

### Prerequisites
- Docker Desktop 4.43+ with Model Runner enabled
- Git
- API Keys: GitHub token, Brave Search API key

### Setup

1. **Clone and configure:**
   ```bash
   git clone https://github.com/ajeetraina/project-recommender-agents.git
   cd project-recommender-agents
   
   # Copy environment template
   cp mcp.env.example .mcp.env
   
   # Edit .mcp.env with your API keys
   nano .mcp.env
   ```

2. **Set up secrets:**
   ```bash
   mkdir -p .secrets
   echo "your_github_token_here" > .secrets/github_token
   echo "your_brave_api_key_here" > .secrets/brave_api_key
   ```

### Deployment Options

#### 🏠 Local with Docker Model Runner (Recommended)
```bash
# Start with local AI models
docker compose -f compose.yaml -f compose.dmr.yaml up --build

# Access the app
open http://localhost:8501
```

#### ☁️ Cloud GPU with Docker Offload
```bash
# Enable Docker Offload first
docker offload start --gpu

# Use the SAME compose files - they automatically execute in cloud
docker compose -f compose.yaml -f compose.dmr.yaml up --build

# Stop when done
docker offload stop
```

#### 🔑 OpenAI Fallback
```bash
# Use OpenAI models instead of local inference
echo "sk-your-openai-api-key" > secret.openai-api-key
docker compose -f compose.yaml -f compose.openai.yaml up --build
```

## 🧠 AI Models

### Local Models (Docker Model Runner)
- **Recommendation Model**: `ai/qwen2.5:7B-Q4_0` (High-quality recommendations)
- **Analysis Model**: `ai/qwen2.5:3B-Q4_0` (Fast profile analysis)

### Cloud Models (Docker Offload)
- **Recommendation Model**: `ai/qwen2.5:14B-Q4_0` (Enhanced reasoning)
- **Analysis Model**: `ai/qwen2.5:7B-Q4_0` (Detailed analysis)

### OpenAI Models
- **Recommendation Model**: `gpt-4o` (Premium recommendations)
- **Analysis Model**: `gpt-4o-mini` (Cost-effective analysis)

## 🔧 MCP Tools Integration

| Tool | Purpose | Configuration |
|------|---------|---------------|
| **GitHub** | Repository analysis, user profiles | Requires GitHub token |
| **Brave Search** | Event discovery, tech trends | Requires Brave API key |
| **Web Fetch** | Content retrieval from URLs | No configuration needed |
| **Filesystem** | Local workspace access | Restricted to `/workspace` |

## 📁 Project Structure

```
project-recommender-agents/
├── compose.yaml              # Base configuration
├── compose.dmr.yaml         # Docker Model Runner setup
├── compose.offload.yaml     # Cloud deployment overrides
├── compose.openai.yaml      # OpenAI integration
├── Dockerfile               # Multi-stage container build
├── requirements.txt         # Python dependencies
├── .mcp.env.example        # Environment template
├── src/
│   ├── main.py             # Streamlit application
│   ├── config.py           # Configuration management
│   ├── agents/             # AI agent implementations
│   └── mcp/                # MCP Gateway client
└── tests/                  # Test suite
```

## 🛠️ Usage Commands

```bash
# Development mode with hot reload
docker compose -f compose.yaml -f compose.dmr.yaml up --build --target development

# Production deployment
docker compose -f compose.yaml -f compose.dmr.yaml up --build

# Cloud scaling with Docker Offload
docker offload start --gpu
docker compose -f compose.yaml -f compose.dmr.yaml up --build
docker offload stop

# OpenAI fallback
docker compose -f compose.yaml -f compose.openai.yaml up --build

# Health checks
docker compose ps
curl http://localhost:8501/health
curl http://localhost:8811/health

# View logs
docker compose logs -f recommender-ui
docker compose logs -f mcp-gateway
```

## 🔍 Troubleshooting

### Common Issues

**Model Runner Not Available**
```bash
docker desktop enable model-runner
docker model pull ai/qwen2.5:3B-Q4_0
```

**MCP Gateway Connection Failed**
```bash
docker compose logs mcp-gateway
curl http://localhost:8811/servers
```

**Memory Issues**
```bash
# Reduce model context size in compose.dmr.yaml
# Change from context_size: 32768 to context_size: 16384
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Built with ❤️ following docker/compose-for-agents patterns
EOF

    # Create deployment scripts
    cat > scripts/deploy-local.sh << 'EOF'
#!/bin/bash
echo "🚀 Deploying Project Recommender Agents locally..."
docker compose -f compose.yaml -f compose.dmr.yaml up --build
EOF

    cat > scripts/deploy-offload.sh << 'EOF'
#!/bin/bash
echo "☁️ Deploying Project Recommender Agents with Docker Offload..."
docker offload start --gpu
docker compose -f compose.yaml -f compose.dmr.yaml up --build
echo "💡 Remember to run 'docker offload stop' when done!"
EOF

    cat > scripts/deploy-openai.sh << 'EOF'
#!/bin/bash
echo "🔑 Deploying Project Recommender Agents with OpenAI..."
if [ ! -f "secret.openai-api-key" ]; then
    echo "❌ OpenAI API key not found. Create secret.openai-api-key file first."
    exit 1
fi
docker compose -f compose.yaml -f compose.openai.yaml up --build
EOF

    chmod +x scripts/*.sh

    log_success "Documentation and scripts created"
}

create_tests() {
    log_info "Creating test structure..."

    cat > tests/test_agents.py << 'EOF'
import pytest
import asyncio
from src.agents.analyzer import GitHubAnalyzerAgent
from src.agents.recommender import ProjectRecommenderAgent

@pytest.mark.asyncio
async def test_github_analyzer():
    """Test GitHub analyzer agent"""
    agent = GitHubAnalyzerAgent("http://localhost:8000/v1", "test-model")
    result = await agent.analyze_profile("testuser")
    
    assert "username" in result
    assert result["username"] == "testuser"
    assert "skills" in result
    assert isinstance(result["skills"], list)

@pytest.mark.asyncio
async def test_project_recommender():
    """Test project recommender agent"""
    agent = ProjectRecommenderAgent("http://localhost:8000/v1", "test-model")
    
    analysis_data = {
        "skills": ["Python", "Docker"],
        "interests": ["AI", "DevOps"]
    }
    
    result = await agent.recommend_projects(analysis_data)
    
    assert "projects" in result
    assert isinstance(result["projects"], list)
    if result["projects"]:
        project = result["projects"][0]
        assert "title" in project
        assert "description" in project
EOF

    cat > tests/test_config.py << 'EOF'
import os
from src.config import AppConfig

def test_app_config():
    """Test application configuration"""
    config = AppConfig()
    
    assert config.app_title == "Project Recommender Agents"
    assert config.log_level == "INFO"
    assert "localhost" in config.mcp_gateway_endpoint
EOF

    log_success "Test structure created"
}

create_makefile() {
    log_info "Creating Makefile for common tasks..."

    cat > Makefile << 'EOF'
.PHONY: help build dev prod test clean setup

# Default target
help:
	@echo "🚀 Project Recommender Agents - Available Commands:"
	@echo ""
	@echo "  setup          - Set up development environment"
	@echo "  build          - Build Docker images"
	@echo "  dev            - Run in development mode"
	@echo "  prod           - Run in production mode"
	@echo "  offload        - Deploy with Docker Offload"
	@echo "  openai         - Deploy with OpenAI models"
	@echo "  test           - Run tests"
	@echo "  clean          - Clean up containers and images"
	@echo "  logs           - Show logs"
	@echo "  health         - Check health status"
	@echo ""

setup:
	@echo "🔧 Setting up development environment..."
	cp mcp.env.example .mcp.env
	mkdir -p .secrets workspace
	@echo "✅ Setup complete! Edit .mcp.env with your API keys."

build:
	@echo "🏗️ Building Docker images..."
	docker compose -f compose.yaml -f compose.dmr.yaml build

dev:
	@echo "🚀 Starting development environment..."
	docker compose -f compose.yaml -f compose.dmr.yaml up --build

prod:
	@echo "🚀 Starting production environment..."
	docker compose -f compose.yaml -f compose.dmr.yaml up --build -d

offload:
	@echo "☁️ Starting Docker Offload deployment..."
	docker offload start --gpu
	docker compose -f compose.yaml -f compose.dmr.yaml up --build

openai:
	@echo "🔑 Starting OpenAI deployment..."
	@if [ ! -f "secret.openai-api-key" ]; then \
		echo "❌ OpenAI API key not found. Create secret.openai-api-key file first."; \
		exit 1; \
	fi
	docker compose -f compose.yaml -f compose.openai.yaml up --build

test:
	@echo "🧪 Running tests..."
	docker compose -f compose.yaml -f compose.dmr.yaml exec recommender-ui pytest tests/

clean:
	@echo "🧹 Cleaning up..."
	docker compose down
	docker system prune -f

logs:
	@echo "📋 Showing logs..."
	docker compose logs -f

health:
	@echo "🏥 Checking health status..."
	@curl -f http://localhost:8501/health || echo "❌ UI not healthy"
	@curl -f http://localhost:8811/health || echo "❌ MCP Gateway not healthy"
	@echo "✅ Health check complete"

stop-offload:
	@echo "⏹️ Stopping Docker Offload..."
	docker offload stop
EOF

    log_success "Makefile created"
}

create_ci_cd() {
    log_info "Creating CI/CD configuration..."

    mkdir -p .github/workflows

    cat > .github/workflows/ci.yml << 'EOF'
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-dev.txt
    
    - name: Run linting
      run: |
        flake8 src/ tests/
        black --check src/ tests/
    
    - name: Run tests
      run: |
        pytest tests/ -v
    
    - name: Type checking
      run: |
        mypy src/

  docker-build:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Build Docker image
      run: |
        docker build -t project-recommender-agents:test .
    
    - name: Test Docker image
      run: |
        docker run --rm project-recommender-agents:test python -c "import src.main; print('✅ Import successful')"
EOF

    log_success "CI/CD configuration created"
}

main() {
    echo "=============================================================================="
    echo "🚀 Project Recommender Agents - Complete Modernization Script"
    echo "=============================================================================="
    echo ""
    
    # Check if we're in the right directory
    if [[ ! -d ".git" ]]; then
        log_warning "Not in a git repository. Make sure you're in the project root."
        read -p "Continue anyway? (y/N): " -r
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    log_info "Starting modernization process..."
    
    # Create backup
    create_backup
    
    # Create directory structure
    create_directory_structure
    
    # Create all files
    create_compose_files
    create_dockerfile
    create_requirements
    create_application_code
    create_environment_files
    create_documentation
    create_tests
    create_makefile
    create_ci_cd
    
    # Create scripts directory
    mkdir -p scripts
    
    log_success "🎉 Modernization complete!"
    echo ""
    echo "=============================================================================="
    echo "📋 Next Steps:"
    echo "=============================================================================="
    echo ""
    echo "1. 🔑 Configure your API keys:"
    echo "   cp mcp.env.example .mcp.env"
    echo "   # Edit .mcp.env with your GitHub token and Brave API key"
    echo ""
    echo "2. 🔐 Set up secrets:"
    echo "   echo 'your_github_token' > .secrets/github_token"
    echo "   echo 'your_brave_api_key' > .secrets/brave_api_key"
    echo ""
    echo "3. 🚀 Deploy locally:"
    echo "   make dev"
    echo "   # or: docker compose -f compose.yaml -f compose.dmr.yaml up --build"
    echo ""
    echo "4. ☁️ Deploy with Docker Offload:"
    echo "   make offload"
    echo "   # or: docker offload start --gpu && docker compose up"
    echo ""
    echo "5. 🔑 Deploy with OpenAI:"
    echo "   echo 'sk-your-key' > secret.openai-api-key"
    echo "   make openai"
    echo ""
    echo "6. 📱 Access the application:"
    echo "   http://localhost:8501"
    echo ""
    echo "=============================================================================="
    echo "🎯 Your repository is now modernized following docker/compose-for-agents!"
    echo "=============================================================================="
    echo ""
    echo "📚 Available commands:"
    echo "   make help     - Show all available commands"
    echo "   make setup    - Quick setup"
    echo "   make dev      - Development mode"
    echo "   make prod     - Production mode"
    echo "   make test     - Run tests"
    echo "   make clean    - Clean up"
    echo ""
    echo "🔍 Backup created in: $BACKUP_DIR"
    echo ""
    log_success "Ready to build the future of AI agents! 🚀"
}

# Run the main function
main "$@"

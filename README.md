# 🚀 Project Recommender Agents - Modernized

A multi-agent system that analyzes GitHub profiles to recommend personalized hackathon projects and relevant events, following the latest Docker Compose AI patterns.

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




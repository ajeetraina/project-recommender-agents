# Project Recommender Agents

A multi-agent system that analyzes GitHub profiles to recommend personalized hackathon projects and relevant events.

## Quick Start

1. **Setup environment:**
   ```bash
   cp mcp.env.example .mcp.env
   # Edit .mcp.env with your API keys
   ```

2. **Run with local model:**
   ```bash
   docker compose up --build
   ```

3. **Access the app:** http://localhost:8501

## Alternative Configurations

**OpenAI models:**
```bash
echo "sk-your-api-key" > secret.openai-api-key
docker compose -f compose.yaml -f compose.openai.yaml up
```

**Docker Offload (cloud GPU):**
```bash
docker compose -f compose.yaml -f compose.offload.yaml up
```

## MCP Servers Used

- **GitHub** - Repository analysis and user data
- **Brave Search** - Web search capabilities  
- **Web Fetch** - Content retrieval from URLs

## Model

- **Local:** `ai/qwen2.5:3b` via Docker Model Runner
- **OpenAI:** `gpt-4o-mini` (with override)
- **Offload:** `ai/qwen2.5:7b` (cloud GPU)

## Architecture

```
Streamlit UI (:8501) → MCP Servers → LLM
```

The system uses a simple Streamlit interface with MCP server selection, similar to Claude Desktop, allowing users to:
- Select active MCP servers
- View available tools
- Analyze GitHub profiles
- Get personalized project recommendations

## Requirements

- Docker Desktop 4.43+ with Model Runner enabled
- API keys: GitHub token, Brave Search API key

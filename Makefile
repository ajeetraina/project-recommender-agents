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

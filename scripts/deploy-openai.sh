#!/bin/bash
echo "🔑 Deploying Project Recommender Agents with OpenAI..."
if [ ! -f "secret.openai-api-key" ]; then
    echo "❌ OpenAI API key not found. Create secret.openai-api-key file first."
    exit 1
fi
docker compose -f compose.yaml -f compose.openai.yaml up --build

#!/bin/bash
echo "🚀 Deploying Project Recommender Agents locally..."
docker compose -f compose.yaml -f compose.dmr.yaml up --build

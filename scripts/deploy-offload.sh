#!/bin/bash
echo "☁️ Deploying Project Recommender Agents with Docker Offload..."
docker offload start --gpu
docker compose -f compose.yaml -f compose.dmr.yaml up --build
echo "💡 Remember to run 'docker offload stop' when done!"

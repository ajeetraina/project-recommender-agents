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

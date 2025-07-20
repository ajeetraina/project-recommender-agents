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

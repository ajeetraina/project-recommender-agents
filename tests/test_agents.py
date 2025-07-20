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

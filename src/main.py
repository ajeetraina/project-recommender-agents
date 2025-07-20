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

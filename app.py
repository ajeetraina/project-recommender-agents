import streamlit as st
import requests
import json
import os
from typing import Dict, List, Any

# Configure page
st.set_page_config(
    page_title="Project Recommender Agents",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'mcp_servers' not in st.session_state:
    st.session_state.mcp_servers = {}
if 'selected_servers' not in st.session_state:
    st.session_state.selected_servers = ['github', 'brave-search']

def load_mcp_config():
    """Load MCP server configuration"""
    try:
        with open('.mcp.env', 'r') as f:
            env_vars = {}
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    env_vars[key] = value
        return env_vars
    except FileNotFoundError:
        return {}

def get_available_mcp_servers():
    """Get list of available MCP servers"""
    return {
        'github': {
            'name': 'GitHub',
            'description': 'Access GitHub repositories and user data',
            'tools': ['get_user', 'list_repos', 'get_repo_contents', 'search_repos'],
            'required_env': 'GITHUB_TOKEN'
        },
        'brave-search': {
            'name': 'Brave Search',
            'description': 'Web search capabilities',
            'tools': ['web_search'],
            'required_env': 'BRAVE_API_KEY'
        },
        'fetch': {
            'name': 'Web Fetch',
            'description': 'Fetch content from URLs',
            'tools': ['fetch_url'],
            'required_env': None
        }
    }

def analyze_github_user(username: str, selected_servers: List[str]) -> Dict[str, Any]:
    """Analyze GitHub user using selected MCP servers"""
    
    # Mock implementation for demo
    # In real implementation, this would use MCP clients
    
    analysis = {
        'username': username,
        'repositories': [
            {
                'name': 'awesome-web-app',
                'language': 'JavaScript',
                'description': 'A modern web application',
                'stars': 45,
                'technologies': ['React', 'Node.js', 'Express']
            },
            {
                'name': 'ml-experiments',
                'language': 'Python', 
                'description': 'Machine learning experiments',
                'stars': 23,
                'technologies': ['Python', 'TensorFlow', 'Pandas']
            },
            {
                'name': 'mobile-app',
                'language': 'TypeScript',
                'description': 'Cross-platform mobile app',
                'stars': 67,
                'technologies': ['React Native', 'TypeScript']
            }
        ],
        'languages': {'JavaScript': 3, 'Python': 2, 'TypeScript': 1},
        'focus_areas': ['Web Development', 'Data Science', 'Mobile Development'],
        'experience_level': 'Intermediate'
    }
    
    return analysis

def generate_recommendations(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate project recommendations based on analysis"""
    
    recommendations = [
        {
            'title': 'AI-Powered Code Review Assistant',
            'category': 'Developer Tools',
            'description': 'Build an intelligent code review tool using machine learning',
            'technologies': ['Python', 'TensorFlow', 'Git APIs'],
            'difficulty': 'Medium',
            'estimated_hours': 35,
            'relevance_score': 92
        },
        {
            'title': 'Real-time Collaboration Platform',
            'category': 'Web Development',
            'description': 'Create a modern real-time collaboration tool',
            'technologies': ['React', 'Node.js', 'WebSockets'],
            'difficulty': 'Medium',
            'estimated_hours': 28,
            'relevance_score': 88
        },
        {
            'title': 'Mobile Health Tracker',
            'category': 'Mobile Apps',
            'description': 'Develop a cross-platform health and fitness tracker',
            'technologies': ['React Native', 'TypeScript', 'Health APIs'],
            'difficulty': 'Hard',
            'estimated_hours': 42,
            'relevance_score': 85
        }
    ]
    
    return recommendations

# Main UI
st.title("🤖 Project Recommender Agents")
st.markdown("*Get personalized hackathon project recommendations based on your GitHub profile*")

# Sidebar for MCP server configuration
with st.sidebar:
    st.header("🔧 MCP Configuration")
    
    # Load environment variables
    env_vars = load_mcp_config()
    available_servers = get_available_mcp_servers()
    
    st.subheader("Available MCP Servers")
    
    selected_servers = []
    for server_id, server_info in available_servers.items():
        # Check if required environment variable is available
        env_available = (server_info['required_env'] is None or 
                        server_info['required_env'] in env_vars)
        
        status_icon = "✅" if env_available else "❌"
        
        if st.checkbox(
            f"{status_icon} {server_info['name']}", 
            value=(server_id in st.session_state.selected_servers and env_available),
            disabled=not env_available,
            key=f"server_{server_id}"
        ):
            selected_servers.append(server_id)
    
    st.session_state.selected_servers = selected_servers
    
    # Display selected server tools
    if selected_servers:
        st.subheader("Available Tools")
        for server_id in selected_servers:
            server_info = available_servers[server_id]
            with st.expander(f"🔨 {server_info['name']} Tools"):
                for tool in server_info['tools']:
                    st.code(tool)
    
    # Environment status
    st.subheader("Environment Status")
    required_vars = ['GITHUB_TOKEN', 'BRAVE_API_KEY']
    for var in required_vars:
        if var in env_vars:
            st.success(f"✅ {var}")
        else:
            st.error(f"❌ {var}")
    
    if not all(var in env_vars for var in ['GITHUB_TOKEN']):
        st.warning("⚠️ Configure API keys in .mcp.env file")

# Main content area
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📊 User Input")
    
    # User input form
    with st.form("analysis_form"):
        username = st.text_input(
            "GitHub Username", 
            value="octocat",
            help="Enter the GitHub username to analyze"
        )
        
        theme = st.selectbox(
            "Hackathon Theme",
            ["General", "Sustainability", "Healthcare", "FinTech", "Education"],
            help="Choose the hackathon theme for recommendations"
        )
        
        skill_level = st.selectbox(
            "Skill Level",
            ["Beginner", "Intermediate", "Advanced"],
            index=1
        )
        
        submitted = st.form_submit_button("🚀 Analyze & Recommend")
    
    # Display selected MCP servers
    if st.session_state.selected_servers:
        st.subheader("🔗 Active MCP Servers")
        for server_id in st.session_state.selected_servers:
            server_info = available_servers[server_id]
            st.info(f"**{server_info['name']}**: {server_info['description']}")

with col2:
    st.header("📈 Results")
    
    if submitted:
        if not username:
            st.error("Please enter a GitHub username")
        elif not st.session_state.selected_servers:
            st.error("Please select at least one MCP server")
        else:
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Analyze GitHub profile
            status_text.text("🔍 Analyzing GitHub profile...")
            progress_bar.progress(25)
            
            try:
                analysis = analyze_github_user(username, st.session_state.selected_servers)
                
                # Step 2: Generate recommendations
                status_text.text("🎯 Generating recommendations...")
                progress_bar.progress(75)
                
                recommendations = generate_recommendations(analysis)
                
                # Step 3: Complete
                status_text.text("✅ Analysis complete!")
                progress_bar.progress(100)
                
                # Display results
                st.success(f"Analysis completed for **{username}**")
                
                # User profile summary
                with st.expander("👤 Profile Summary", expanded=True):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Experience Level", analysis['experience_level'])
                        st.metric("Total Repositories", len(analysis['repositories']))
                    with col_b:
                        st.metric("Primary Language", list(analysis['languages'].keys())[0])
                        st.metric("Focus Areas", len(analysis['focus_areas']))
                    
                    st.write("**Focus Areas:**", ", ".join(analysis['focus_areas']))
                
                # Project recommendations
                st.subheader("🚀 Recommended Projects")
                
                for i, rec in enumerate(recommendations[:3]):
                    with st.container():
                        st.markdown(f"### {i+1}. {rec['title']}")
                        
                        col_x, col_y, col_z = st.columns([2, 1, 1])
                        with col_x:
                            st.write(rec['description'])
                            st.write(f"**Category:** {rec['category']}")
                        with col_y:
                            st.metric("Relevance", f"{rec['relevance_score']}%")
                            st.write(f"**Difficulty:** {rec['difficulty']}")
                        with col_z:
                            st.metric("Est. Hours", rec['estimated_hours'])
                        
                        # Technology tags
                        st.write("**Technologies:**")
                        tech_cols = st.columns(len(rec['technologies']))
                        for j, tech in enumerate(rec['technologies']):
                            tech_cols[j].code(tech)
                        
                        st.divider()
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                progress_bar.empty()
                status_text.empty()

# Footer
st.markdown("---")
st.markdown("""
**MCP Servers Used:** GitHub, Brave Search, Web Fetch  
**Model:** ai/qwen2.5:3b (via Docker Model Runner)  
**Framework:** Streamlit + Docker Compose for Agents
""")

# Debug info (only in development)
if os.getenv('DEBUG'):
    with st.expander("🐛 Debug Info"):
        st.json({
            'selected_servers': st.session_state.selected_servers,
            'env_vars_loaded': len(load_mcp_config()),
            'model_url': os.getenv('MODEL_URL', 'Not configured')
        })

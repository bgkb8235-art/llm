import os
from typing import Annotated, TypedDict, List, Dict, Any
from typing_extensions import Literal

import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import ArxivAPIWrapper
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END

# Streamlit Page Configuration
st.set_page_config(
    page_title="AI Research Architect",
    page_icon="⚡",
    layout="wide"
)

# Custom CSS for Modern Gradient UI
st.markdown("""
<style>
    /* Glassmorphism Title Box */
    .title-box {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        color: white;
        padding: 24px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.4);
    }
    .title-box h1 {
        color: #ffffff !important;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .title-box p {
        color: #f1f5f9 !important;
        font-size: 1.1rem;
        margin-top: 5px;
    }
    
    /* Sidebar Styling */
    div[data-testid="stSidebar"] {
        background-color: #0f172a;
    }
    
    /* Custom Card Containers */
    .metric-card {
        background-color: #1e293b;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #a855f7;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 1. API KEYS SETUP (SIDEBAR)
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/isometric-line/100/brain.png", width=64)
    st.title("🔑 API Configuration")
    
    groq_api_key = st.text_input("GROQ API Key", type="password", value=os.environ.get("GROQ_API_KEY", ""))
    tavily_api_key = st.text_input("Tavily API Key", type="password", value=os.environ.get("TAVILY_API_KEY", ""))
    
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key
    if tavily_api_key:
        os.environ["TAVILY_API_KEY"] = tavily_api_key

    st.markdown("---")
    st.markdown("### 🛠️ Agents Architecture")
    st.markdown("""
    1. **Explorer Agent**: Searches literature via Tavily & ArXiv.
    2. **Reviewer Agent**: Identifies research gaps.
    3. **Writer Agent**: Drafts the academic paper structure.
    """)


# ==========================================
# 2. STATE & LANGGRAPH AGENT SETUP
# ==========================================

class AgentState(TypedDict):
    """The state of the research assistant."""
    messages: Annotated[list[BaseMessage], lambda x, y: x + y]
    domain: str
    basic_problem: str
    research_gaps: List[str]
    paper_draft: Dict[str, str]
    next_step: str


def build_research_graph():
    """Initializes tools and compiles the LangGraph state graph."""
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_retries=2
    )
    search_tool = TavilySearchResults(max_results=3)
    arxiv_tool = ArxivAPIWrapper()

    def create_agent(llm, tools, system_prompt: str):
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])
        if tools:
            return prompt | llm.bind_tools(tools)
        return prompt | llm

    def explorer_node(state: AgentState):
        explorer_prompt = f"""You are the Search Specialist using GROQ speed.
        Domain: {state['domain']}
        Current Problem: {state['basic_problem']}

        Task: Find the 3 most relevant current research directions/papers.
        Focus on specific methodologies and architectural trends."""

        agent = create_agent(llm, [search_tool], explorer_prompt)
        response = agent.invoke(state)
        return {"messages": [response], "next_step": "reviewer"}

    def reviewer_node(state: AgentState):
        reviewer_prompt = f"""You are the Literature Reviewer.
        Analyze gathered info and identify 3 critical 'Research Gaps'.
        What is missing in {state['domain']} regarding {state['basic_problem']}?"""

        agent = create_agent(llm, None, reviewer_prompt)
        response = agent.invoke(state)

        gaps = [line for line in response.content.split('\n') if 'gap' in line.lower() or '-' in line][:3]
        if not gaps:
            gaps = [
                "Gap 1: Limited real-time adaptability under sudden dynamic changes.",
                "Gap 2: High computational latency in sub-second inference.",
                "Gap 3: Lack of generalized edge-device evaluation metrics."
            ]

        return {
            "messages": [response],
            "research_gaps": gaps,
            "next_step": "writer"
        }

    def writer_node(state: AgentState):
        writer_prompt = f"""You are the Academic Paper Architect.
        Gaps identified: {state['research_gaps']}
        Draft a comprehensive, structured paper containing:
        - Title
        - Abstract
        - Introduction & Literature Context
        - Identified Research Gaps
        - Proposed Methodology & Architecture
        - Conclusion & Future Scope

        Tone: Formal Academic Markdown."""

        agent = create_agent(llm, None, writer_prompt)
        response = agent.invoke(state)

        return {
            "messages": [response],
            "paper_draft": {"full_report": response.content},
            "next_step": "end"
        }

    def router(state: AgentState):
        return state["next_step"]

    # Graph Construction
    workflow = StateGraph(AgentState)
    workflow.add_node("explorer", explorer_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("writer", writer_node)

    workflow.add_edge(START, "explorer")
    workflow.add_conditional_edges("explorer", router, {"reviewer": "reviewer", "end": END})
    workflow.add_conditional_edges("reviewer", router, {"writer": "writer", "end": END})
    workflow.add_edge("writer", END)

    return workflow.compile()


# ==========================================
# 3. STREAMLIT UI & RUNNER
# ==========================================

# Main Header Title Box
st.markdown("""
    <div class="title-box">
        <h1>⚡ LangGraph + GROQ AI Research Architect</h1>
        <p>Automated Multi-Agent Academic Exploration, Gap Analysis & Paper Generation</p>
    </div>
""", unsafe_allow_html=True)

col_input, col_output = st.columns([1, 2])

with col_input:
    st.subheader("📥 Input Research Query")
    domain_input = st.text_input(
        "Research Domain",
        value="IoT-Based Clinical Decision Support and AI Healthcare Devices"
    )
    problem_input = st.text_area(
        "Core Problem Statement",
        value="Real-time patient monitoring and Explainable AI risk prediction in Panchakarma therapies using low-cost ESP32 IoT hardware and Firebase cloud infrastructure.",
        height=140
    )
    
    start_btn = st.button("🚀 Run Multi-Agent Analysis", use_container_width=True, type="primary")

with col_output:
    if start_btn:
        if not os.environ.get("GROQ_API_KEY") or not os.environ.get("TAVILY_API_KEY"):
            st.error("⚠️ Please provide both GROQ and Tavily API keys in the sidebar to run the analysis.")
        else:
            try:
                research_app = build_research_graph()
                
                initial_input = {
                    "domain": domain_input,
                    "basic_problem": problem_input,
                    "messages": [HumanMessage(content=f"Analyze research for domain: {domain_input}, problem: {problem_input}")],
                    "research_gaps": [],
                    "paper_draft": {},
                    "next_step": ""
                }

                # Containers for status and real-time step monitoring
                status_container = st.status("🚀 Initializing Multi-Agent System...", expanded=True)
                
                tab1, tab2, tab3 = st.tabs(["📑 Generated Academic Paper", "🔍 Agent 1: Search Specialist", "📊 Agent 2: Gap Reviewer"])
                
                with tab1:
                    paper_placeholder = st.empty()
                with tab2:
                    explorer_placeholder = st.empty()
                with tab3:
                    reviewer_placeholder = st.empty()
                    gaps_placeholder = st.empty()

                # Stream through graph steps
                for output in research_app.stream(initial_input, stream_mode="updates"):
                    for node_name, node_state in output.items():
                        
                        if node_name == "explorer":
                            status_container.update(label="🔍 Search Specialist: Gathering papers...", state="running")
                            last_msg = node_state["messages"][-1].content
                            explorer_placeholder.markdown(f"### 🔍 Exploratory Search Summary\n\n{last_msg}")

                        elif node_name == "reviewer":
                            status_container.update(label="📊 Literature Reviewer: Analyzing research gaps...", state="running")
                            last_msg = node_state["messages"][-1].content
                            reviewer_placeholder.markdown(f"### 📊 Literature Review Summary\n\n{last_msg}")
                            
                            gaps = node_state.get("research_gaps", [])
                            if gaps:
                                formatted_gaps = "### 🎯 Identified Gaps:\n" + "\n".join([f"- **Gap {i+1}:** {gap.strip()}" for i, gap in enumerate(gaps)])
                                gaps_placeholder.markdown(formatted_gaps)

                        elif node_name == "writer":
                            status_container.update(label="📑 Paper Architect: Drafting final paper...", state="running")
                            paper_content = node_state["paper_draft"].get("full_report", "Draft generation failed.")
                            paper_placeholder.markdown(paper_content)

                status_container.update(label="✅ Analysis Complete!", state="complete", expanded=False)

            except Exception as e:
                st.error(f"Execution Error: {e}")
    else:
        st.info("Enter your parameters on the left and click **Run Multi-Agent Analysis** to begin.")

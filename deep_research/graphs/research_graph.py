from langgraph.graph import StateGraph,START,END
from deep_research.states.state_research import ResearcherOutputState,ResearcherState
from deep_research.nodes.research_node import llm_call,tool_node,compress_research,should_continue


def get_research_workflow():
    """Alias for get_scope_research_workflow."""
    # Build the agent workflow
    agent_builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)
    # Add nodes to the graph
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node)
    agent_builder.add_node("compress_research", compress_research)

    # Add edges to connect nodes
    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges(
        "llm_call",
        should_continue,
        {
            "tool_node": "tool_node", # Continue research loop
            "compress_research": "compress_research", # Provide final answer
        },
    )
    agent_builder.add_edge("tool_node", "llm_call") # Loop back for more research
    agent_builder.add_edge("compress_research", END)

    # Compile the agent
    researcher_agent = agent_builder.compile()
    return researcher_agent
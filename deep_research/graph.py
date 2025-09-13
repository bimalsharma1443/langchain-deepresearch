from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.memory import InMemorySaver

from deep_research.states.state_research import ResearcherOutputState,ResearcherState
from deep_research.states.state_scope import AgentState,AgentInputState
from deep_research.nodes.scope_node import clarify_with_user,write_research_brief
from deep_research.nodes.research_node import llm_call,tool_node,compress_research,should_continue


def get_scope_research_workflow():
    """Construct and return the scope research workflow graph."""
    # build workflow graph
    workflow = StateGraph(AgentState,input_schema=AgentInputState)

    # add node to workflow
    workflow.add_node("clarify_with_user",clarify_with_user)
    workflow.add_node("write_research_brief",write_research_brief)

    # add edge to workflow
    workflow.add_edge(START,"clarify_with_user")
    workflow.add_edge("clarify_with_user","write_research_brief")
    workflow.add_edge("write_research_brief",END)

    # checkpointer
    checkpointer = InMemorySaver()

    # compile workflow
    scope_research = workflow.compile(checkpointer=checkpointer)
    return scope_research

def get_reseaerchh_workflow():
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

    agent_builder.add_edge("tool_node", "llm_call") # Loop back for more research
    agent_builder.add_edge("compress_research", END)

    # Compile the agent
    researcher_agent = agent_builder.compile()
    return researcher_agent
    
from langgraph.graph import StateGraph,START,END

from deep_research.nodes.scope_node import clarify_with_user,write_research_brief
from deep_research.nodes.report_generation_node import final_report_generation
from deep_research.states.state_scope import AgentState,AgentInputState
from deep_research.graphs.research_graph import get_research_workflow


def get_full_workflow():
    """Construct and return the full research workflow graph."""
    # Build the agent workflow
    deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

    # Add nodes to the graph
    # Add workflow nodes
    deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
    deep_researcher_builder.add_node("write_research_brief", write_research_brief)
    deep_researcher_builder.add_node("supervisor_subgraph", get_research_workflow())
    deep_researcher_builder.add_node("final_report_generation", final_report_generation)

    # Add edges to connect nodes
    # Add workflow edges
    deep_researcher_builder.add_edge(START, "clarify_with_user")
    deep_researcher_builder.add_edge("write_research_brief", "supervisor_subgraph")
    deep_researcher_builder.add_edge("supervisor_subgraph", "final_report_generation")
    deep_researcher_builder.add_edge("final_report_generation", END)

    # Compile the full workflow
    agent = deep_researcher_builder.compile()
    return agent
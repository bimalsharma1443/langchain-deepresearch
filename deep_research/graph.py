from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.memory import InMemorySaver

from deep_research.states.state_research import ResearcherOutputState,ResearcherState
from deep_research.states.state_supervisor import SupervisorState
from deep_research.states.state_scope import AgentState,AgentInputState
from deep_research.nodes.scope_node import clarify_with_user,write_research_brief
from deep_research.nodes.supervisor_node import supervisor,supervisor_tools
from deep_research.mcp import llm_call as mcp_llm_call,tool_node as mcp_tool_node,compress_research as mcp_compress_research,should_continue as mcp_should_continue


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

def get_research_mcp_workflow():
    """Construct and return the research MCP workflow graph."""
    # Build the agent workflow
    agent_builder_mcp = StateGraph(ResearcherState, output_schema=ResearcherOutputState)

    # Add nodes to the graph
    agent_builder_mcp.add_node("llm_call", mcp_llm_call)
    agent_builder_mcp.add_node("tool_node", mcp_tool_node)
    agent_builder_mcp.add_node("compress_research", mcp_compress_research)

    # Add edges to connect nodes
    agent_builder_mcp.add_edge(START, "llm_call")
    agent_builder_mcp.add_conditional_edges(
        "llm_call",
        mcp_should_continue,
        {
            "tool_node": "tool_node",        # Continue to tool execution
            "compress_research": "compress_research",  # Compress research findings
        },
    )
    agent_builder_mcp.add_edge("tool_node", "llm_call")  # Loop back for more processing
    agent_builder_mcp.add_edge("compress_research", END)

    agent_mcp = agent_builder_mcp.compile()
    return agent_mcp

def get_supervisor_workflow():

    supervisor_builder = StateGraph(SupervisorState)
    supervisor_builder.add_node("supervisor", supervisor)
    supervisor_builder.add_node("supervisor_tools", supervisor_tools)
    supervisor_builder.add_edge(START, "supervisor")
    supervisor_agent = supervisor_builder.compile()
    return supervisor_agent

from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.memory import InMemorySaver

from deep_research.states.state_scope import AgentState,AgentInputState
from deep_research.nodes.node import clarify_with_user,write_research_brief

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
from langchain_core.messages import HumanMessage
from deep_research.util import create_thread
from deep_research.graph import get_scope_research_workflow,get_research_mcp_workflow,get_supervisor_workflow
from deep_research.graphs.research_graph import get_research_workflow
from deep_research.graphs.full_graph import get_full_workflow
from deep_research.util import save_scope_image

graph = None  # Global variable to hold the scope object

def set_scope():
    """Initialize the global graph variable."""
    global graph
    graph = get_scope_research_workflow()

def set_research():
    """Initialize the global graph variable."""
    global graph
    graph = get_research_workflow()

def set_research_mcp():
    """Initialize the global graph variable."""
    global graph
    graph = get_research_mcp_workflow()

def set_supervisor():
    """Initialize the global graph variable."""
    global graph
    graph = get_supervisor_workflow()

def set_full_report():
    """Initialize the global graph variable."""
    global graph
    graph = get_full_workflow()

def scope_invoke(content: str):
    """Invoke the scope with a message."""
    if graph is None:
        raise ValueError("Scope is not initialized. Call set_scope() first.")
    config = create_thread()
    result = graph.invoke({"messages": [HumanMessage(content=content)]}, config=config)
    return result

def research_invoke(research_brief: str):
    """Invoke the scope with a message."""
    if graph is None:
        raise ValueError("Scope is not initialized. Call set_scope() first.")
    result = graph.invoke({"researcher_messages": [HumanMessage(content=f"{research_brief}.")]})
    return result

async def research_mcp_invoke(research_brief: str):
    """Invoke the scope with a message."""
    if graph is None:
        raise ValueError("Scope is not initialized. Call set_scope() first.")
    result = await graph.ainvoke({"researcher_messages": [HumanMessage(content=f"{research_brief}.")]})
    return result

async def supervisor_invoke(research_brief: str):
    """Invoke the scope with a message."""
    if graph is None:
        raise ValueError("Scope is not initialized. Call set_scope() first.")
    result = await graph.ainvoke({"supervisor_messages": [HumanMessage(content=f"{research_brief}.")]})
    return result

async def full_report_invoke(content: str):
    """Invoke the scope with a message."""
    thread = {"configurable": {"thread_id": "1", "recursion_limit": 50}}
    if graph is None:
        raise ValueError("Scope is not initialized. Call set_scope() first.")
    result = await graph.ainvoke({"messages": [HumanMessage(content=content)]}, config=thread)
    return result

def save_image(filename: str = "scope_graph.png"):
    """Save the graphical representation of the scope to a PNG file."""
    if graph is None:
        raise ValueError("Scope is not initialized. Call set_scope() first.")
    save_scope_image(graph, filename)
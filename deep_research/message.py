from langchain_core.messages import HumanMessage
from deep_research.util import create_thread
from deep_research.graph import get_scope_research_workflow
from deep_research.util import save_scope_image

scope = None  # Global variable to hold the scope object

def set_scope():
    """Initialize the global scope variable."""
    global scope
    scope = get_scope_research_workflow()

def invoke(content: str):
    """Invoke the scope with a message."""
    if scope is None:
        raise ValueError("Scope is not initialized. Call set_scope() first.")
    config = create_thread()
    result = scope.invoke({"messages": [HumanMessage(content=content)]}, config=config)
    return result

def save_image(filename: str = "scope_graph.png"):
    """Save the graphical representation of the scope to a PNG file."""
    if scope is None:
        raise ValueError("Scope is not initialized. Call set_scope() first.")
    save_scope_image(scope)
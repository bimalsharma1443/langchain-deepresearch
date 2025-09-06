"""State definitions for research agent scopes.
"""

import operator
from typing_extensions import Optional, Annotated, List, Sequence

from langchain_core.messages import BaseMessage

from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages


# Define a states
class AgentInputState(MessagesState):
    """Input state for the full agent - only contains messages from user input."""
    pass

class AgentState(MessagesState):
    """
    Main state for the full multi-agent research system.
    
    Extends MessagesState with additional fields for research coordination.
    Note: Some fields are duplicated across different state classes for proper
    state management between subgraphs and the main workflow.
    """
    # Research brief generated from user conversation history
    research_brief: Optional[str]
    # Message exchange with supervisor agent for co-ordination
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    # Raw unprocessed research notes collected during the research phase
    raw_notes: Annotated[List[str],operator.add] = []
    # Processed and structured notes ready for report generation
    notes: Annotated[List[str],operator.add] = []
    # Final formatted research report
    final_report: str



import operator
from typing_extensions import TypedDict,Annotated,Sequence,List

from langchain_core.messages import BaseMessage

class ResearcherState(TypedDict):
    """
    State for the research agent containing message history and research metadata.
    
    This state tracks the researcher's conversation, iteration count for limiting
    tool calls, the research topic being investigated, compressed findings,
    and raw research notes for detailed analysis.
    """
    researcher_messages: Annotated[Sequence[BaseMessage],operator.add]
    tool_call_iteration: int
    research_topic: str
    compressed_researcher: str
    raw_notes: Annotated[List[str],operator.add]

class ResearcherOutputState(TypedDict):
    """
    Output state for the research agent containing final research results.
    
    This represents the final output of the research process with compressed
    research findings and all raw notes from the research process.
    """
    compressed_researcher: str
    raw_notes: Annotated[List[str],operator.add]
    researcher_messages: Annotated[Sequence[BaseMessage],operator.add]

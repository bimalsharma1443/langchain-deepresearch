from langchain_core.messages import HumanMessage,AIMessage,get_buffer_string
from langgraph.types import Command
from langgraph.graph import END

from deep_research.states.state_scope import AgentState
from deep_research.states.schemas import ClarifyWithUser,ResearchQuestion
from deep_research.prompt.prompts import clarify_with_user_instructions
from deep_research.model import initialize_ollama_model
from deep_research.util import get_today_str

def clarify_with_user(state: AgentState) -> bool:
    """
    Determine if the user's request contains sufficient information to proceed with research.

    Uses structured output to make deterministic decisions and avoid hallucination.
    Routes to either research brief generation or ends with a clarification question.
    """
    # initialize model
    model = initialize_ollama_model()
    # set up structured output model
    structured_output_model = model.with_structured_output(ClarifyWithUser)
    # invoke a model with stuctured clarification instructions
    response = structured_output_model.invoke([
        HumanMessage(content=clarify_with_user_instructions.format(
            messages=get_buffer_string(messages=state["messages"]), 
            date=get_today_str()
        ))
    ])

    # round based on clarification needed
    if response.need_clarification:
        return Command(
            goto=END,
            update={"messages": [AIMessage(content=response.question)]},
        )
    else:
        return Command(
            goto="write_research_brief", 
            update={"messages": [AIMessage(content=response.verification)]}
        )
    

def write_research_brief(state: AgentState):
    """
    Transform the conversation history into a comprehensive research brief.
    
    Uses structured output to ensure the brief follows the required format
    and contains all necessary details for effective research.
    """
    # initialize model
    model = initialize_ollama_model()
    # set up structured output model
    structured_output_model = model.with_structured_output(ResearchQuestion)
    # invoke a model with stuctured clarification instructions
    response = structured_output_model.invoke([
        HumanMessage(content=clarify_with_user_instructions.format(
            messages=get_buffer_string(messages=state["messages"]), 
            date=get_today_str()
        ))
    ])

    # Update state with generated research brief and pass it to the supervisor
    return {
        "research_brief": response.research_brief,
        "supervisor_messages": [HumanMessage(content=f"{response.research_brief}.")]
    }

    



    
    

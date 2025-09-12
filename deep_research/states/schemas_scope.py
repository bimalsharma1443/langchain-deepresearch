from pydantic import Field,BaseModel
from typing_extensions import List


class ClarifyWithUser(BaseModel):
    """Schema for user clarification decisions and questions."""
    need_clarification: bool = Field(..., description="Whether the user needs to be asked a clarifying question.")
    question: str = Field("", description="A question asked the user to clarify the report scope.")
    verification: str = Field("", description="Verify message that we will start research after the user has provided the necessary information.")

class ResearchQuestion(BaseModel):
    """Schema for structured research brief generation."""
    research_brief: List[str] = Field(..., description="A researcher question that will be used to guide a research.")
from pydantic import Field,BaseModel

class Summary(BaseModel):
    """Schema for webpage content summarization."""
    summary: str = Field(..., description="Concise summary of the webpage content.")
    key_excerpts: str = Field(..., description="Important quotes and excerpts from the content.")
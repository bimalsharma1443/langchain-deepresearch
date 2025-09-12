from deep_research.tools import tavily_search, think_tool
from deep_research.model import initialize_ollama_model

tools = [tavily_search, think_tool]

tools_by_name = {tool.name: tool for tool in tools}

model = initialize_ollama_model()

model_with_tools = model.bind_tools(tools)




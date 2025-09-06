1. install uv `https://docs.astral.sh/uv/getting-started/installation/`
2. un init . to create new project but we will do `uv sync --all-extras --dev`
3. `source .venv/bin/activate` activate
4. add following projects. `uv add langgraph langchain langchain_community langchain_tavily  langchain_mcp_adapters pydantic rich tavily-python`
5. add ollama `langchain-ollama`
6. inroduction and installation about ollama. 
   1. https://ollama.com/download
   2. https://github.com/ollama/ollama
7. run this to run ap roject if you get erro non module named. `export PYTHONPATH=/Users/bimal.sharma/llm/langchain-deepresearch:$PYTHONPATH`

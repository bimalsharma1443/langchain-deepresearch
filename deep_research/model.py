from langchain_ollama.chat_models import ChatOllama 

def initialize_ollama_model():
    """Initialize and return the Ollama chat model."""
    
    chat_model = ChatOllama(model="llama3.1:8b",temperature=0)
    return chat_model
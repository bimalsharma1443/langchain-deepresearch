from unittest import result
from dotenv import load_dotenv
from deep_research.message import invoke, set_scope, save_image

def main():
    print("Hello from langchain-deepresearch!")
    print("Loading environment variables and initializing workflow...")
    load_dotenv() # Load environment variables from .env file
    set_scope()  # Initialize the workflow scope
    save_image("scope_graph.png")  # Save the graphical representation of the scope
    print("Workflow initialized. Here's the graphical representation:")
    result = invoke("I want to research the best coffee shops in San Francisco.")
    for message in result["messages"]:
        message.pretty_print()




if __name__ == "__main__":
    main()

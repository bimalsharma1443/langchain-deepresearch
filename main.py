from unittest import result
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
from deep_research.message import research_invoke, set_scope, save_image,set_research,research_mcp_invoke,set_research_mcp

import asyncio

async def main():
    print("Hello from langchain-deepresearch!")
    print("Loading environment variables and initializing workflow...")
    
    # scope
    # set_scope()  # Initialize the workflow scope
    # save_image("scope_graph.png")  # Save the graphical representation of the scope
    # print("Workflow initialized. Here's the graphical representation:")
    # result = invoke("Planning to go Mumbai in October from Varanasi.")
    # for message in result["messages"]:
    #     message.pretty_print()

    # research
    set_research_mcp()  # Initialize the workflow scope
    save_image("research_graph.png")  # Save the graphical representation of the scope
    research_brief = """I want to identify and evaluate the coffee shops in San Francisco that are considered the best based specifically  
    on coffee quality. My research should focus on analyzing and comparing coffee shops within the San Francisco area, 
    using coffee quality as the primary criterion. I am open regarding methods of assessing coffee quality (e.g.,      
    expert reviews, customer ratings, specialty coffee certifications), and there are no constraints on ambiance,      
    location, wifi, or food options unless they directly impact perceived coffee quality. Please prioritize primary    
    sources such as the official websites of coffee shops, reputable third-party coffee review organizations (like     
    Coffee Review or Specialty Coffee Association), and prominent review aggregators like Google or Yelp where direct  
    customer feedback about coffee quality can be found. The study should result in a well-supported list or ranking of
    the top coffee shops in San Francisco, emphasizing their coffee quality according to the latest available data as  
    of July 2025."""

    # result = research_invoke(research_brief)
    result = await research_mcp_invoke(research_brief)
    from rich.markdown import Markdown
    print(result)
    md = Markdown(result['compressed_research'])
    print(md.markup)

if __name__ == "__main__":
    asyncio.run(main())

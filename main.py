from unittest import result
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
from deep_research.message import research_invoke, set_scope, save_image,set_research,research_mcp_invoke,set_research_mcp,set_supervisor,set_full_report,full_report_invoke

import asyncio

async def main():
    print("Hello from langchain-deepresearch!")
    print("Loading environment variables and initializing workflow...")

    set_full_report()  # Initialize the workflow scope
    save_image("full_report.png")  # Save the graphical representation of the scope
    research_brief = """Compare Gemini to OpenAI Deep Research agents."""

    # result = research_invoke(research_brief)
    result = await full_report_invoke(research_brief)
    print("==="*30)
    for message in result['messages']:
        message.pretty_print()

if __name__ == "__main__":
    asyncio.run(main())

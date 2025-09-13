from langchain_core.tools import tool, InjectedToolArg
from typing_extensions import Annotated, List, Literal

from deep_research.util import tavily_search_multiple,dublicate_search_results,process_search_results,format_search_output


@tool(parse_docstring=True,error_on_invalid_docstring=False)
def tavily_search(
    query: str, 
    max_results: Annotated[int, InjectedToolArg] = 3,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
) -> List[dict]:
    """Fetch results from Tavily search API with content summarization.

    Args:
        query: A single search query to execute
        max_results: Maximum number of results to return
        topic: Topic to filter results by ('general', 'news', 'finance')

    Returns:
        Formatted string of search results with summaries
    """
    search_result = tavily_search_multiple(
        search_queries=[query],
        max_results=max_results,
        topic=topic,
        include_raw_content=True
    )

    # Deduplicate results by URL to avoid processing duplicate content
    unique_results = dublicate_search_results(search_result)

    # Process results with summarization
    summarized_results = process_search_results(unique_results)

    # Format output for consumption
    return format_search_output(summarized_results)
from datetime import datetime

def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %-d, %Y")

def save_scope_image(scope, filename: str = "scope_graph.png"):
    """Save the graphical representation of the scope to a PNG file."""
    with open(filename, "wb") as f:
        f.write(scope.get_graph(xray=True).draw_mermaid_png())
    print(f"Image saved as {filename}")

def create_thread(thread_id: str = "1") -> dict:
    thread = {"configurable": {"thread_id": thread_id}}
    return thread

def format_search_output(summarized_results: dict) -> str:
    """Format search results into a well-structured string output.

    Args:
        summarized_results: Dictionary of processed search results

    Returns:
        Formatted string of search results with clear source separation
    """
    if not summarized_results:
        return "No valid search results found. Please try different search queries or use a different search API."

    formatted_output = "Search results: \n\n"

    for i, (url, result) in enumerate(summarized_results.items(), 1):
        formatted_output += f"\n\n--- SOURCE {i}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "-" * 80 + "\n"

    return formatted_output
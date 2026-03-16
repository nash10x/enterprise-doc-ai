from tavily import TavilyClient

from config import TAVILY_API_KEY, TAVILY_SEARCH_DOMAIN


def search_web(query: str, max_results: int = 5) -> str:
    """
    Search the official OpenText Access Manager documentation via Tavily.
    Returns formatted context string from web results.
    """
    if not TAVILY_API_KEY:
        return ""

    client = TavilyClient(api_key=TAVILY_API_KEY)

    response = client.search(
        query=query,
        include_domains=[TAVILY_SEARCH_DOMAIN],
        max_results=max_results,
        search_depth="advanced",
    )

    results = response.get("results", [])
    if not results:
        return ""

    snippets = []
    for r in results:
        snippets.append(f"[{r['title']}]({r['url']})\n{r['content']}")

    return "\n\n---\n\n".join(snippets)

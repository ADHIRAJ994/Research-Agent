from tavily import TavilyClient


def make_web_search_tool(api_key: str):
    client = TavilyClient(api_key=api_key)

    def web_search(query: str, max_results: int = 4) -> list[dict]:
        response = client.search(query=query, max_results=max_results)
        return [
            {"title": r["title"], "url": r["url"], "content": r["content"]}
            for r in response["results"]
        ]

    return web_search


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    web_search = make_web_search_tool(os.getenv("TAVILY_API_KEY"))
    for r in web_search("pricing model of Notion AI"):
        print(f"- {r['title']} ({r['url']})")
        print(f"  {r['content'][:120]}...\n")
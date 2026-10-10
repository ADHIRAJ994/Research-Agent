from tavily import TavilyClient
import os
from dotenv import load_dotenv
load_dotenv()
def make_web_search_tool(api_key:str):
    """Returns a web_search function bound to a tavily client."""
    client = TavilyClient(api_key=api_key)

    def web_search(query:str,max_results:int=4)->list[dict]:
        response = client.search(query=query,max_results=max_results)
        return[
            {"title": r["title"], "url": r["url"], "content": r["content"]}
            for r in response["results"]
        ]

    return web_search

## Testing the function
TAVILY_API_KEY= os.getenv('TAVILY_API_KEY')
web_search = make_web_search_tool(TAVILY_API_KEY)
results = web_search("pricing model of Notion AI")
for r in results:
    print(f"- {r['title']} ({r['url']})")
    print(f"  {r['content'][:120]}...\n")
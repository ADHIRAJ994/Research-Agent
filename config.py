import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "./data/index")
MAX_REVISIONS = int(os.environ.get("MAX_REVISIONS", 2))

assert GROQ_API_KEY, "GROQ_API_KEY not set — check your .env file."
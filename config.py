import os
from dotenv import load_dotenv
load_dotenv()
# Explicitly load .env from the project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

FAISS_INDEX_PATH = os.getenv(
    "FAISS_INDEX_PATH",
    "./data/index"
)

MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", 2))

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Check your .env file.")
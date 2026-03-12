"""All configuration — loaded from .env file."""
import os
from dotenv import load_dotenv

load_dotenv()

# LLM provider: "openai" | "groq" | "ollama"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

# Provider credentials and model names
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL     = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3")

# Local embedding model — no API key needed
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Number of chunks to retrieve per query
TOP_K = 4

# Path to contract documents folder
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

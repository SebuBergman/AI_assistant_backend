import os
import tiktoken

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pymilvus import MilvusClient

from app.db.database import lifespan

load_dotenv()

APP_NAME = "AI Agent API"
APP_DESCRIPTION = "AI Agent API with Milvus, PostgreSQL, and Redis"
APP_VERSION = "1.0.8"

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

UPLOAD_PATH = "./data"
os.makedirs(UPLOAD_PATH, exist_ok=True)

# Zilliz Cloud / Milvus config
ZILLIZ_CLOUD_URI = os.getenv("ZILLIZ_CLOUD_URI")
ZILLIZ_CLOUD_TOKEN = os.getenv("ZILLIZ_CLOUD_TOKEN")

# Initialize Milvus client for Zilliz Cloud
milvus_client = MilvusClient(
    uri=ZILLIZ_CLOUD_URI,
    token=ZILLIZ_CLOUD_TOKEN
)

# Initialize embeddings (text-embedding-3-small is 1536 dimensions)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
_TOKEN_ENCODER = tiktoken.encoding_for_model("text-embedding-3-small")
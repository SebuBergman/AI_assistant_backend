import spacy
import boto3
import os
from openai import OpenAI
from anthropic import Anthropic
from langchain_openai import ChatOpenAI
from app.config import OPENAI_API_KEY, DEEPSEEK_API_KEY, ANTHROPIC_API_KEY
import warnings

from app.db.vectorstore_manager import get_vectorstore

# Suppress spaCy W108 warnings
warnings.filterwarnings("ignore", message=".*W108.*")

# Clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# NLP
nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")

# Vectorstore
vectorstore = get_vectorstore()

# S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)

# Backend – AI Assistant API

## 📌 Overview
This backend powers an AI-driven assistant with support for:
- Multi-model text generation (13 LLMs including GPT-5.2, Claude Sonnet 4.5, DeepSeek)
- Implemented tools (2): Tavily Search and OpenWeather APIs
- Email rewriting & generation
- RAG with hybrid keyword + vector search
- PDF uploading, embedding, querying, and retrieval
- Model capabilities discovery (incl. tool support)
- Milvus/Zilliz statistics & schema validation
- Token usage tracking & reporting
- Input tokens, output tokens, and total token counts are calculated and reported at the end of each full assistant run (after all prompts/steps are completed).
- RAG transparency (context + references)
- RAG responses include the retrieved context and/or references used by the assistant to generate its answer (e.g., quoted snippets, document titles etc.).

---

## 🧠 Tech Stack
- **Python**
- **FastAPI**
- **OpenAI / Anthropic / DeepSeek APIs**
- **LangChain / LangChain Core**
- **Pydantic**
- **dotenv**
- **Milvus / Zilliz cloud (pymilvus)**
- **boto3**
- **Tavily**
- **RESTful API Design**

---

## 🚀 Getting Started

### 1. Clone the repo:
```bash
git clone https://github.com/your-username/AI_assistant_backend.git
cd backend-repo
```

### 2. Create virtual environment
```bash
uv venv
```

### 3. Install dependencies
```bash
uv pip install -r requirements.txt
```

### 4. Environment variables
Create a `.env` file based on `.env-template`:

# Copy everything from .env-template and fill in your credentials
```
# API key for Tavily service (Web search)
TAVILY_API_KEY=<your-key>
# API keys for AI services
OPENAI_API_KEY=<your-key>
ANTHROPIC_API_KEY=<your-key>
DEEPSEEK_API_KEY=<your-key>

# Zilliz cloud config
ZILLIZ_CLOUD_URI=<public-endpoint>
ZILLIZ_CLOUD_TOKEN=<your-token>
EMBEDDING_DIM=<embedding-dimension (1536 for current setup)>

# Cluster and collection names
DATABASE_NAME=<cluster-name>
UPLOAD_PATH=<name-of-local-upload-directory (e.g. "./data")>
QUERY_CACHE_COLLECTION_NAME=<query-cache collection name>
EMBEDDINGS_COLLECTION_NAME=<embeddings collection name>
PDFS_COLLECTION_NAME=<pdf collection name>

# AWS S3 Configuration
AWS_ACCESS_KEY_ID=<your-key>
AWS_SECRET_ACCESS_KEY=<your-key>
AWS_REGION=<your-region>
S3_BUCKET_NAME=<bucket-name>
...
```

**Steps**
1. Create `.env` file in the root  
2. Copy values from `.env-template`  
3. Replace `<placeholder>` with your real credentials  

---

### 5. Create local folder
Windows / Linux / macOS:
```powershell
mkdir <UPLOAD_PATH>
```

### 6. Run locally
Windows:
```powershell
.venv\Scripts\activate
uvicorn main:app --reload
```

Linux / macOS:
```powershell
source .venv/bin/activate
uvicorn main:app --reload
```

## Or run with wsl
### Step 1: Open WSL (Ubuntu)
```powershell
wsl
```

### Step 2: Build from inside WSL
Navigate to your backend inside WSL’s filesystem:
```powershell
cd ~/projects/AI_assistant_backend
docker build -t assistant_backend .
```

### Step 3: Run you backend
```powershell
docker run --env-file .env -p 8000:8000 assistant_backend
```

---

## 📡 API Endpoints

### **Email Assistant**
#### `POST /email_assistant`
Rewrite or generate email text based on user prompt.

---

### **Unified Generator**
#### `POST /api/generate`
- Streamed response generation  
- Select from **13 LLM models**  
- Adjustable temperature  
- Optional **RAG**:  
  - Keyword (optional) + vector hybrid search  
  - Uses embedded PDFs  
  - Upload and choose PDF sources  

---

### **Vector Database**
#### `POST /query`
Query embedding db via hybrid search (keyword (optional) + vector).

---

### **Model Management**
#### `GET /api/models`
List available LLM models.

#### `GET /api/tools/{model_name}`
Check if the chosen model supports **tool calling**.

---

### **PDF Embeddings**
#### `POST /upload`
Upload & embed a PDF.

#### `GET /fetch_pdfs`
Retrieve list of available PDFs.

---

### **Caching System**
#### `/cache/stats`
Cache statistics

#### `/cache/clear_old`
Clear expired cache entries (older than 30 days)

#### `/cache/clear_all`
Clear all cached content

#### `/cache/entries`
List all cached entries

---

### **System Utilities**
#### `/clear_all`
Remove:
- embeddings  
- PDFs  
- S3 content  

#### `/milvus/stats`
See Milvus/Zilliz collection stats.

#### `/milvus/schema`
View & verify current Milvus collection schema.

#### `/health`
Health check endpoint.

---

## 📄 .env Template
Make sure to check the `.env-template` file inside the repository.  
All sensitive authentication keys go there (rename to .env to not accidentally upload to Github).

---

## 🛠️ Other Notes
- Structured and modular FastAPI architecture  
- Ready for Dockerization  
- Integrates directly with the frontend for streaming AI responses
- Create requirements.txt with command: uv pip freeze > requirements.txt

---

## 🔗 Frontend Link
👉 **View the Frontend README here:**  
[`https://github.com/your-username/AI_assistant_frontend`](https://github.com/SebuBergman/AI_assistant_frontend)

---

## 📝 TODO
- Add Email rewrite usage analytics
- Fix caching
- Add more tools for the AI assistant to use
# Backend Repository (FastAPI + Python)<br>
📌 Project Name: GenAI Email Assistant & AI Chat Backend<br><br>
🚀 Description:<br>
FastAPI-powered backend for email tone transformation (professional, friendly, persuasive) and a multi-model AI assistant. Features advanced prompt engineering, OpenAI integration, and dynamic model selection.

🛠️ Tech Stack:<br>
Python + FastAPI<br>
OpenAI API<br>
RESTful API Design

🧰 Getting Started:<br>
‼️ Prerequisites: Python 3.9+ and venv

Clone the repo:
```bash
git clone https://github.com/your-username/backend-repo.git
cd backend-repo
```

Set up a virtual environment:
```bash
uv venv
```

Install dependencies:
```bash
uv pip install -r requirements.txt
```

Create .env (requirements)
TAVILY_API_KEY=<tavily_key>

Run locally:
```bash
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
uvicorn main:app --reload
```

→ API docs at http://localhost:8000/docs

🔗 Frontend Integration: Configure the frontend to point to http://localhost:8000

<a href="https://github.com/SebuBergman/AI_assistant_frontend">AI front-end</a>

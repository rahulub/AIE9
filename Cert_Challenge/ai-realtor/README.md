# AI Realtor

A streaming AI chat application for home inspection analysis with a React frontend and FastAPI backend.

## Project Structure

```
ai-realtor/
├── backend/
│   ├── main.py              # FastAPI app entry point
│   ├── routers/
│   │   └── chat.py          # /api/chat streaming endpoint
│   ├── requirements.txt
│   └── .env.example         # Copy to .env and add your API keys
└── frontend/
    ├── public/
    │   └── index.html
    └── src/
        ├── App.jsx
        ├── components/
        │   ├── ChatInput.jsx
        │   └── ChatResponse.jsx
        └── hooks/
            └── useStreamingChat.js   # Core streaming logic (TODO)
```

## Getting Started

### Backend

```bash
cd backend

# Install dependencies (uv creates .venv automatically)
uv sync

# For local development:
cp .env.example .env.local   # edit with your local API keys

# For production/web deployment:
cp .env.example .env         # edit with your production API keys

# Run the server
uv run uvicorn main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm start
```

## What to implement

- [ ] `backend/routers/chat.py` — wire up LLM streaming in `stream_llm_response()`
- [ ] `frontend/src/hooks/useStreamingChat.js` — implement the streaming fetch loop
- [ ] `frontend/src/App.jsx` — connect `handleSubmit` to `sendMessage`
- [ ] `frontend/src/components/ChatInput.jsx` — add Enter key submit
- [ ] Future: PDF upload endpoint + extract text before sending to LLM

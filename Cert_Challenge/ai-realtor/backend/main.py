from pathlib import Path
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import logging

_base_dir = Path(__file__).resolve().parent
load_dotenv(_base_dir / ".env")
load_dotenv(_base_dir / ".env.local", override=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import chat, ingest
from routers.ingest import auto_ingest_data_dir

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs auto-ingestion of backend/data/ PDFs before the server accepts requests."""
    await auto_ingest_data_dir()
    yield


app = FastAPI(title="LLM Chat API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api")
app.include_router(ingest.router, prefix="/api")


@app.get("/health")
def health_check():
    return {"status": "ok"}

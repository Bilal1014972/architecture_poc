import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

app = FastAPI(title="Recipe AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PromptInput(BaseModel):
    prompt: str


class PromptResponse(BaseModel):
    response: str


@app.get("/")
async def root():
    return {"status": "running"}


@app.post("/ask", response_model=PromptResponse)
async def ask(data: PromptInput):
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            res = await client.post(
                f"{OLLAMA_URL}/api/chat",
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "application/json",
                },
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [{"role": "user", "content": data.prompt}],
                    "stream": False,
                },
            )
            res.raise_for_status()
            return PromptResponse(response=res.json()["message"]["content"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
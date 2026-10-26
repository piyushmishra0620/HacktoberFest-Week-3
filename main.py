"""FastAPI backend: upload a file -> extract -> summarise with Gemma via Ollama (streamed)."""
import asyncio, base64, json, os
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from extractors import extract

OLLAMA = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
DEFAULT_MODEL = os.getenv("MODEL", "gemma3:4b")      # gemma3 4b+ is multimodal
NUM_CTX = int(os.getenv("NUM_CTX", 8192))
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", 12000))
MAX_MB = int(os.getenv("MAX_UPLOAD_MB", 100))
MAX_IMAGES = int(os.getenv("MAX_IMAGES", 8))

STYLES = {
    "concise": "Write a concise summary of 5-8 sentences.",
    "bullets": "Write a bulleted summary: key points, important numbers, and conclusions.",
    "detailed": "Write a detailed, well-structured summary with headings for each major section.",
    "executive": "Write an executive summary: purpose, key findings, risks, and recommended next steps.",
}

app = FastAPI(title="DocSum")


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def opts():
    return {"num_ctx": NUM_CTX, "temperature": 0.2}


async def chat(client, model, messages) -> str:
    r = await client.post(f"{OLLAMA}/api/chat", json={
        "model": model, "messages": messages, "stream": False, "options": opts()})
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


async def chat_stream(client, model, messages):
    async with client.stream("POST", f"{OLLAMA}/api/chat", json={
            "model": model, "messages": messages, "stream": True, "options": opts()}) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if line:
                d = json.loads(line)
                tok = d.get("message", {}).get("content", "")
                if tok:
                    yield tok


def chunk(text: str, size: int):
    out, cur = [], ""
    for para in text.split("\n"):
        while len(para) > size:                      # hard-split very long lines
            out.append(para[:size]); para = para[size:]
        if len(cur) + len(para) + 1 > size and cur:
            out.append(cur); cur = ""
        cur += para + "\n"
    if cur.strip():
        out.append(cur)
    return out


def instruction(style, language, focus):
    s = STYLES.get(style, STYLES["concise"])
    f = f" Pay special attention to: {focus}." if focus.strip() else ""
    return f"{s} Write in {language}.{f} Only use information from the document; do not invent facts."


@app.get("/api/models")
async def models():
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{OLLAMA}/api/tags")
            names = [m["name"] for m in r.json().get("models", [])]
        return {"models": names, "default": DEFAULT_MODEL, "ollama": True}
    except Exception:
        return {"models": [DEFAULT_MODEL], "default": DEFAULT_MODEL, "ollama": False}


@app.post("/api/summarize")
async def summarize(file: UploadFile = File(...), model: str = Form(DEFAULT_MODEL),
                    style: str = Form("concise"), language: str = Form("English"),
                    focus: str = Form("")):
    data = await file.read()
    if len(data) > MAX_MB * 1024 * 1024:
        return JSONResponse({"error": f"File larger than {MAX_MB} MB"}, status_code=413)
    name = file.filename or "upload"

    async def gen():
        try:
            yield sse("status", f"Reading {name}…")
            doc = await asyncio.to_thread(extract, name, data)
            for n in doc.notes:
                yield sse("note", n)
            if not doc.text.strip() and not doc.images:
                yield sse("error", "No readable content found in this file.")
                return

            async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=10)) as client:
                text = doc.text
                # 1) describe visuals with the vision-capable model
                for i, img in enumerate(doc.images[:MAX_IMAGES], 1):
                    yield sse("status", f"Looking at image {i} of {min(len(doc.images), MAX_IMAGES)}…")
                    desc = await chat(client, model, [{
                        "role": "user",
                        "content": "Describe this image for a document summary. Transcribe any visible "
                                   "text, and explain charts, tables or diagrams precisely.",
                        "images": [base64.b64encode(img).decode()]}])
                    text += f"\n\n[Visual {i}: {desc}]"

                # 2) map-reduce for long documents
                parts = chunk(text, CHUNK_CHARS)
                task = instruction(style, language, focus)
                if len(parts) > 1:
                    partials = []
                    for i, p in enumerate(parts, 1):
                        yield sse("status", f"Summarising section {i} of {len(parts)}…")
                        partials.append(await chat(client, model, [{
                            "role": "user",
                            "content": f"Summarise this part of a larger document, keeping key facts, "
                                       f"figures and names:\n\n{p}"}]))
                    body = "\n\n".join(f"Section {i}:\n{s}" for i, s in enumerate(partials, 1))
                    prompt = (f"Below are summaries of consecutive sections of one document "
                              f"('{name}'). Combine them into one final summary. {task}\n\n{body}")
                else:
                    prompt = f"Summarise the document '{name}'. {task}\n\n{parts[0]}"

                yield sse("status", "Writing summary…")
                async for tok in chat_stream(client, model, [{"role": "user", "content": prompt}]):
                    yield sse("token", tok)
            yield sse("done", "ok")
        except httpx.ConnectError:
            yield sse("error", f"Cannot reach Ollama at {OLLAMA}. Run `ollama serve`.")
        except httpx.HTTPStatusError as e:
            yield sse("error", f"Ollama error {e.response.status_code}. Is the model pulled? "
                               f"Try `ollama pull {model}`.")
        except Exception as e:
            yield sse("error", str(e))

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")

app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

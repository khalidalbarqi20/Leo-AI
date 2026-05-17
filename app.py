"""
Leo-AI — Production-grade coding assistant backend.
Architecture: FastAPI + httpx async streaming + SSE.
"""

import os
import base64
import logging
import re
import zipfile
import tarfile
import io
import json
import urllib.parse
import asyncio
import httpx

from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, Response, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("leo-ai")

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Leo-AI", version="3.0.0")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY")

# Per-session state
chat_sessions: dict[str, list] = {}
stop_flags:    dict[str, asyncio.Event] = {}

# ─── SYSTEM PROMPT (senior engineer persona) ──────────────────────────────────
SYSTEM_PROMPT = """You are a senior software engineer, system architect, debugging specialist, security reviewer, and production-grade AI coding agent.

Your responsibility is NOT merely generating code.
Your responsibility is to engineer reliable, maintainable, scalable, secure, production-quality software systems.

You must behave like an experienced technical lead responsible for the long-term health, stability, maintainability, and architecture quality of the entire project.

==================================================
CORE ENGINEERING MINDSET

- Think before modifying code.
- Analyze before implementing.
- Verify before claiming success.
- Prioritize correctness over speed.
- Prioritize maintainability over shortcuts.
- Treat every change as production-critical.
- Behave like a real senior engineer reviewing mission-critical systems.

Your objective is NOT merely "working code".
Your objective is: Reliable systems, Stable architecture, Minimal regressions, Safe implementations, Scalable code, Long-term maintainability, Production-quality engineering.

==================================================
PROJECT UNDERSTANDING RULES

1. Always understand the ENTIRE project before modifying anything.
   Analyze architecture, dependencies, imports, routing, APIs, database flow, business logic, shared utilities, async flows, environment configs, and state management.

2. Before making changes, summarize: Current architecture, Related systems, Risk areas, Possible regressions, Safer implementation options.

3. Maintain architectural consistency. Respect existing project structure. Keep naming conventions consistent. Reuse existing utilities.

4. Before writing new code: Search for existing related logic. Avoid reinventing existing systems.

==================================================
TASK EXECUTION RULES

5. For medium or large tasks: First create a structured implementation plan. Break work into phases. Then implement carefully step-by-step.

6. Separate concerns properly. Separate bug fixes from refactoring. Keep unrelated code untouched.

7. Prefer minimal, targeted, safe modifications. Never rewrite entire files unless absolutely necessary.

==================================================
DEBUGGING & ERROR PREVENTION

8. Root-cause analysis is mandatory. Identify the actual root cause before fixing. Do not patch symptoms only. Trace execution flow carefully.

9. Regression prevention is critical. Before changing any code: Identify impacted files, predict what could break, maintain backward compatibility.

10. Think defensively. Always anticipate: Edge cases, Null states, Async timing issues, API failures, Race conditions, Scaling problems.

11. Before finalizing any change: Re-check imports, syntax, dependencies, routes, API integrations, environment variables, edge cases, compatibility.

==================================================
CODE QUALITY RULES

12. Generate production-grade code only. Code must be: Clean, Modular, Readable, Maintainable, Scalable, Reusable, Secure, Efficient, Predictable.

13. Prefer long-term maintainability over short-term hacks. Avoid dirty fixes, fragile logic, unnecessary complexity, silent failures.

14. Refactor proactively when justified. Improve weak architecture safely. Reduce duplication. Simplify complex logic.

15. Prefer explicit logic over clever fragile shortcuts.

==================================================
SECURITY & RELIABILITY

16. Security is mandatory. Never expose secrets. Prevent insecure patterns. Validate inputs. Avoid unsafe code execution.

17. Reliability matters more than speed. Never sacrifice stability. Never fake success.

==================================================
QUALITY CONTROL

18. Do not guess. Do not hallucinate. Do not fabricate APIs or functions. Do not hide uncertainty.

19. Never claim success without verification.

20. If requirements are unclear: Ask precise technical questions first.

==================================================
COMMUNICATION

21. Be technically honest. Explain WHY the issue happened, WHY the solution is safer, possible side effects, impacted components.

22. Communication style: Direct, Technical, Precise, Concise but complete.

==================================================
LANGUAGE & FORMAT

- Always respond in Arabic.
- Write all code inside Markdown code blocks.
- For multi-file projects use: ### filename.ext before each code block.
- For complex tasks: show implementation plan first, then code.
- Flag security issues, architectural risks, and potential regressions explicitly.
- When you spot a bug or bad pattern in the user's code, point it out proactively.
"""

# ─── MODELS (quality-focused, fallback chain) ─────────────────────────────────
# Priority: quality > speed. Ordered by code quality for non-vision tasks.
MODELS: list[dict] = [
    {
        "id":      "qwen/qwen3-coder-480b-a35b-instruct:free",
        "name":    "Qwen3 Coder 480B",
        "desc":    "الأقوى للكود • 262K context • متخصص للبرمجة",
        "badge":   "🥇",
        "context": "262K",
        "speed":   "متوسط",
        "tag":     "أفضل للكود",
        "vision":  False,
        "strength": "code",
    },
    {
        "id":      "deepseek/deepseek-r1:free",
        "name":    "DeepSeek R1",
        "desc":    "تفكير عميق • يعرض سلسلة التفكير • ممتاز للـ debugging",
        "badge":   "🧠",
        "context": "128K",
        "speed":   "بطيء",
        "tag":     "تفكير عميق",
        "vision":  False,
        "strength": "reasoning",
    },
    {
        "id":      "moonshotai/kimi-k2:free",
        "name":    "Kimi K2",
        "desc":    "مصمم لبناء المشاريع الكاملة • multi-agent • Python/Rust/Go",
        "badge":   "🚀",
        "context": "128K",
        "speed":   "متوسط",
        "tag":     "بناء مشاريع",
        "vision":  False,
        "strength": "projects",
    },
    {
        "id":      "qwen/qwen2.5-vl-72b-instruct:free",
        "name":    "Qwen2.5 VL 72B",
        "desc":    "يقرأ الصور • الأقوى vision مجاناً",
        "badge":   "👁️",
        "context": "128K",
        "speed":   "متوسط",
        "tag":     "يرى الصور",
        "vision":  True,
        "strength": "vision",
    },
    {
        "id":      "qwen/qwen2.5-vl-32b-instruct:free",
        "name":    "Qwen2.5 VL 32B",
        "desc":    "يقرأ الصور • سريع",
        "badge":   "🔍",
        "context": "128K",
        "speed":   "سريع",
        "tag":     "يرى الصور",
        "vision":  True,
        "strength": "vision",
    },
    {
        "id":      "google/gemma-4-31b-it:free",
        "name":    "Gemma 4 31B",
        "desc":    "يقرأ الصور • 256K context • من Google",
        "badge":   "🌟",
        "context": "256K",
        "speed":   "متوسط",
        "tag":     "يرى الصور",
        "vision":  True,
        "strength": "vision",
    },
    {
        "id":      "openai/gpt-oss-20b:free",
        "name":    "GPT-OSS 20B",
        "desc":    "سريع ودقيق",
        "badge":   "⚡",
        "context": "128K",
        "speed":   "سريع",
        "tag":     "سريع",
        "vision":  False,
        "strength": "general",
    },
    {
        "id":      "meta-llama/llama-4-scout:free",
        "name":    "Llama 4 Scout",
        "desc":    "الأسرع استجابةً",
        "badge":   "⚡",
        "context": "128K",
        "speed":   "سريع جداً",
        "tag":     "الأسرع",
        "vision":  False,
        "strength": "general",
    },
]

VISION_IDS:  list[str] = [m["id"] for m in MODELS if m["vision"]]
CODING_IDS:  list[str] = [m["id"] for m in MODELS if not m["vision"]]

# ─── Markdown → HTML ──────────────────────────────────────────────────────────
def markdown_to_html(text: str) -> str:
    code_blocks: list[tuple[str, str]] = []

    def extract_code(match: re.Match) -> str:
        lang = match.group(1) or "text"
        code = match.group(2)
        idx  = len(code_blocks)
        code_blocks.append((lang, code))
        return f"___CODE_{idx}___"

    text = re.sub(r"```(\w+)?\n(.*?)```", extract_code, text, flags=re.DOTALL)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"`([^`]+)`",        r"<code>\1</code>",         text)
    text = re.sub(r"^### (.*?)$",  r"<h3>\1</h3>",  text, flags=re.MULTILINE)
    text = re.sub(r"^## (.*?)$",   r"<h2>\1</h2>",  text, flags=re.MULTILINE)
    text = re.sub(r"^# (.*?)$",    r"<h1>\1</h1>",  text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.*?)\*",     r"<em>\1</em>",         text)
    text = re.sub(r"^> (.*?)$",   r"<blockquote>\1</blockquote>", text, flags=re.MULTILINE)
    text = re.sub(r"^\d+\.\s+(.*?)$", r"<li>\1</li>", text, flags=re.MULTILINE)
    text = re.sub(r"^[-*]\s+(.*?)$",  r"<li>\1</li>", text, flags=re.MULTILINE)

    result = []
    for p in text.split("\n\n"):
        p = p.strip()
        if p:
            p = p.replace("\n", "<br>")
            if not p.startswith("<"):
                p = f"<p>{p}</p>"
            result.append(p)
    text = "\n".join(result)

    for idx, (lang, code) in enumerate(code_blocks):
        escaped = code.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        b64     = base64.b64encode(code.encode()).decode()
        display = lang if lang else "code"
        html = (
            f'<div class="cw2"><div class="ch">'
            f'<span>{display}</span>'
            f'<button onclick="cpy(this,\'{b64}\')">&#128203; نسخ</button>'
            f'</div><pre><code>{escaped}</code></pre></div>'
        )
        text = text.replace(f"___CODE_{idx}___", html)
    return text

# ─── File extraction ───────────────────────────────────────────────────────────
_FILE_PATTERN = re.compile(
    r"###\s*(\S+\.(?:py|html|css|js|json|txt|md|jsx|ts|tsx|vue|php|"
    r"java|cpp|c|go|rs|swift|kt|dart|rb|sh|sql|xml|yaml|yml))\s*\n*"
    r"```(?:\w+)?\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

def extract_files(text: str) -> dict[str, str]:
    files = {fname.strip(): code.strip() for fname, code in _FILE_PATTERN.findall(text)}
    if not files:
        m = re.search(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
        if m:
            files[_guess_name(m.group(1))] = m.group(1).strip()
    return files

def _guess_name(content: str) -> str:
    cl = content.lower().strip()
    if cl.startswith(("<!doctype", "<html")):           return "index.html"
    if "fastapi" in cl or "flask" in cl or cl.startswith("import "): return "app.py"
    if cl.startswith(("const ", "let ")) or "function " in cl: return "script.js"
    if "body {" in cl or "{" in cl and "color" in cl:  return "style.css"
    if cl.startswith(("{", "[")):                       return "data.json"
    return "code.txt"

# ─── Compressed file reader ────────────────────────────────────────────────────
MAX_FILE_CHARS = 12_000

def read_file_content(filename: str, raw: bytes, content_type: str) -> str:
    name = filename.lower()

    if name.endswith(".zip") or "zip" in content_type:
        parts = [f"[ZIP: {filename}]"]
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                for info in zf.infolist():
                    if info.is_dir(): continue
                    try:
                        parts.append(f"\n--- {info.filename} ---\n"
                                     f"{zf.read(info.filename).decode('utf-8', errors='replace')[:4000]}")
                    except Exception:
                        parts.append(f"\n--- {info.filename} --- (binary)")
        except Exception as e:
            parts.append(f"(failed: {e})")
        return "\n".join(parts)[:MAX_FILE_CHARS]

    if name.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2")) or "tar" in content_type:
        parts = [f"[TAR: {filename}]"]
        try:
            with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
                for m in tf.getmembers():
                    if not m.isfile(): continue
                    try:
                        f = tf.extractfile(m)
                        if f:
                            parts.append(f"\n--- {m.name} ---\n"
                                         f"{f.read().decode('utf-8', errors='replace')[:4000]}")
                    except Exception:
                        parts.append(f"\n--- {m.name} --- (binary)")
        except Exception as e:
            parts.append(f"(failed: {e})")
        return "\n".join(parts)[:MAX_FILE_CHARS]

    try:
        return f"[file: {filename}]\n{raw.decode('utf-8', errors='replace')}"[:MAX_FILE_CHARS]
    except Exception:
        return f"(cannot read: {filename})"

# ─── Image generation (Pollinations) ──────────────────────────────────────────
def _img_url(prompt: str, w: int = 1024, h: int = 1024, model: str = "flux") -> str:
    return (
        f"https://image.pollinations.ai/prompt/{urllib.parse.quote(prompt)}"
        f"?width={w}&height={h}&model={model}&nologo=true&seed={abs(hash(prompt)) % 99999}"
    )

# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {
        "request":       request,
        "models":        MODELS,
        "default_model": MODELS[0]["id"],
    })

@app.get("/models", response_class=JSONResponse)
async def get_models() -> JSONResponse:
    return JSONResponse(MODELS)

# ─── Stop endpoint ────────────────────────────────────────────────────────────
@app.post("/stop")
async def stop_stream() -> JSONResponse:
    ev = stop_flags.get("default")
    if ev:
        ev.set()
    return JSONResponse({"ok": True})

# ─── Streaming chat ────────────────────────────────────────────────────────────
@app.post("/chat-stream")
async def chat_stream(
    request:  Request,
    prompt:   str        = Form(...),
    model_id: str        = Form(None),
    file:     UploadFile = File(None),
) -> StreamingResponse:

    if not OPENROUTER_API_KEY:
        async def _no_key():
            yield f"data: {json.dumps({'error': 'OPENROUTER_API_KEY not set'})}\n\n"
        return StreamingResponse(_no_key(), media_type="text/event-stream")

    # ── Resolve model ──────────────────────────────────────────────────────────
    valid_ids = [m["id"] for m in MODELS]
    chosen    = model_id if model_id in valid_ids else MODELS[0]["id"]

    # ── Process uploaded file ──────────────────────────────────────────────────
    user_text = prompt
    img_part:  dict | None = None
    has_image = False
    switched  = False

    if file and file.filename:
        try:
            raw = await file.read()
            mt  = file.content_type or ""
            if mt.startswith("image/"):
                has_image = True
                b64 = base64.b64encode(raw).decode()
                img_part = {"type": "image_url", "image_url": {"url": f"data:{mt};base64,{b64}"}}
            else:
                user_text = f"{prompt}\n\n{read_file_content(file.filename, raw, mt)}"
        except Exception as exc:
            logger.error(f"File read error: {exc}")

    # ── Auto-switch to vision model if image uploaded ──────────────────────────
    if has_image:
        obj = next((m for m in MODELS if m["id"] == chosen), None)
        if not (obj and obj["vision"]):
            chosen   = VISION_IDS[0] if VISION_IDS else chosen
            switched = True

    # ── Build model fallback chain ─────────────────────────────────────────────
    if has_image:
        ordered = VISION_IDS
    else:
        # Try chosen model first, then rest of coding models as fallback
        ordered = [chosen] + [mid for mid in CODING_IDS if mid != chosen]

    # ── Session ────────────────────────────────────────────────────────────────
    sid = "default"
    chat_sessions.setdefault(sid, [])
    session = chat_sessions[sid]

    # ── Stop event ────────────────────────────────────────────────────────────
    stop_ev = asyncio.Event()
    stop_flags[sid] = stop_ev

    # ── Build messages ─────────────────────────────────────────────────────────
    msg_content = (
        [{"type": "text", "text": user_text}, img_part]
        if img_part else user_text
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(session[-8:])   # last 4 exchanges
    messages.append({"role": "user", "content": msg_content})

    api_headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://leo-ai.app",
        "X-Title":       "Leo-AI",
    }

    # ── Generator ─────────────────────────────────────────────────────────────
    async def streamer():
        full_text  = ""
        used_model = ordered[0] if ordered else chosen
        last_err   = ""
        succeeded  = False

        for mid in ordered:
            if stop_ev.is_set():
                break

            logger.info(f"Trying model: {mid}")
            try:
                timeout = httpx.Timeout(connect=8.0, read=90.0, write=10.0, pool=5.0)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream(
                        "POST",
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=api_headers,
                        json={
                            "model":       mid,
                            "messages":    messages,
                            "temperature": 0.3,   # lower temp → more precise code
                            "max_tokens":  4000,
                            "stream":      True,
                        },
                    ) as resp:
                        # ── Non-200 → try next model ───────────────────────────
                        if resp.status_code != 200:
                            body = await resp.aread()
                            try:
                                err = json.loads(body).get("error", {}).get("message", "")
                            except Exception:
                                err = body.decode(errors="replace")[:200]
                            last_err = f"HTTP {resp.status_code}: {err}"
                            logger.warning(f"{mid} → {last_err}")
                            continue

                        used_model = mid
                        buf = ""
                        got_first = False

                        async for raw_chunk in resp.aiter_text():
                            if stop_ev.is_set():
                                break

                            buf += raw_chunk
                            lines = buf.split("\n")
                            buf   = lines.pop()   # keep incomplete line

                            for line in lines:
                                line = line.strip()
                                if not line.startswith("data:"):
                                    continue
                                ds = line[5:].strip()
                                if ds == "[DONE]":
                                    break
                                try:
                                    delta = json.loads(ds)["choices"][0]["delta"].get("content", "")
                                    if delta:
                                        if not got_first:
                                            got_first = True
                                            # Signal frontend: spinner → streaming
                                            yield f"data: {json.dumps({'first': True})}\n\n"
                                        full_text += delta
                                        yield f"data: {json.dumps({'token': delta})}\n\n"
                                except Exception:
                                    continue

                succeeded = True
                break  # model worked

            except httpx.ReadTimeout:
                last_err = "timeout (model too slow)"
                logger.warning(f"{mid} timed out")
                continue
            except asyncio.CancelledError:
                break
            except Exception as exc:
                last_err = str(exc)
                logger.warning(f"{mid} error: {exc}")
                continue

        # ── Handle stopped / failed ────────────────────────────────────────────
        if stop_ev.is_set() and full_text:
            full_text += "\n\n*(⏹ توقف بواسطة المستخدم)*"
        elif not full_text:
            full_text = f"⚠️ فشلت كل النماذج.\n\nالخطأ الأخير: `{last_err}`\n\nجرب موديلاً آخر أو أعد المحاولة."
            yield f"data: {json.dumps({'token': full_text})}\n\n"

        # ── Persist session ────────────────────────────────────────────────────
        session.append({"role": "user",      "content": prompt})
        session.append({"role": "assistant", "content": full_text})

        used_name = next(
            (m["name"] for m in MODELS if m["id"] == used_model),
            used_model.split("/")[-1],
        )

        yield f"data: {json.dumps({'done': True, 'html': markdown_to_html(full_text), 'files': extract_files(full_text), 'used_model': used_name, 'switched': switched})}\n\n"

    return StreamingResponse(
        streamer(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

# ─── Image generation ──────────────────────────────────────────────────────────
@app.post("/generate-image", response_class=JSONResponse)
async def generate_image(
    prompt: str = Form(...),
    size:   str = Form("1024x1024"),
    model:  str = Form("flux"),
) -> JSONResponse:
    try:
        parts = size.split("x")
        w, h  = (int(parts[0]), int(parts[1])) if len(parts) == 2 else (1024, 1024)
        return JSONResponse({"url": _img_url(prompt, w, h, model), "prompt": prompt})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

# ─── File download ─────────────────────────────────────────────────────────────
_MIME: dict[str, str] = {
    "py":"text/x-python","js":"application/javascript","html":"text/html",
    "css":"text/css","json":"application/json","txt":"text/plain",
    "md":"text/markdown","sh":"text/x-sh","sql":"text/x-sql",
    "xml":"text/xml","yaml":"text/yaml","yml":"text/yaml",
    "ts":"text/typescript","tsx":"text/typescript","jsx":"text/javascript",
}

@app.post("/download")
async def download(filename: str = Form(...), code_content: str = Form(...)) -> Response:
    ext = filename.rsplit(".", 1)[-1].lower()
    return Response(
        content=code_content,
        media_type=_MIME.get(ext, "text/plain"),
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )

# ─── Clear session ─────────────────────────────────────────────────────────────
@app.post("/clear")
async def clear() -> JSONResponse:
    chat_sessions["default"] = []
    ev = stop_flags.get("default")
    if ev:
        ev.set()
    return JSONResponse({"ok": True})

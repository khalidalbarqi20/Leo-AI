"""
Leo-AI v4 — Production coding assistant.
Fixes: multi-attachment (up to 25), correct file download (raw text),
       saved chats sidebar, mobile-first UI.
"""

import os, base64, logging, re, zipfile, tarfile, io, json, urllib.parse, asyncio
import httpx

from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, Response, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from typing import List

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("leo-ai")

app = FastAPI(title="Leo-AI", version="4.0.0")
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY")

chat_sessions: dict[str, list]          = {}
stop_flags:    dict[str, asyncio.Event] = {}

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a senior software engineer, system architect, debugging specialist, security reviewer, and production-grade AI coding agent.

Your responsibility is to engineer reliable, maintainable, scalable, secure, production-quality software systems — not merely generate code.

CORE RULES:
- Think and analyze before writing code.
- Prioritize correctness, maintainability, and security over speed.
- Always identify root cause before fixing bugs.
- Think defensively: edge cases, null states, async issues, race conditions.
- Never expose secrets. Validate all inputs. Prefer safer defaults.
- For complex tasks: show implementation plan first, then code.
- Proactively flag security issues, architectural risks, and regressions.
- Write clean, modular, readable, production-grade code only.
- Prefer explicit logic over clever fragile shortcuts.

FORMAT:
- Always respond in Arabic.
- Write all code inside Markdown code blocks with language tags.
- For multi-file projects: use ### filename.ext before each code block.
- Be concise but complete. No filler text.
"""

# ── Models ─────────────────────────────────────────────────────────────────────
MODELS: list[dict] = [
    {"id":"qwen/qwen3-coder-480b-a35b-instruct:free","name":"Qwen3 Coder 480B","desc":"الأقوى للكود • 262K context","badge":"🥇","context":"262K","speed":"متوسط","tag":"أفضل للكود","vision":False,"strength":"code"},
    {"id":"deepseek/deepseek-r1:free","name":"DeepSeek R1","desc":"تفكير عميق • ممتاز للـ debugging","badge":"🧠","context":"128K","speed":"بطيء","tag":"تفكير عميق","vision":False,"strength":"reasoning"},
    {"id":"moonshotai/kimi-k2:free","name":"Kimi K2","desc":"بناء المشاريع الكاملة • multi-agent","badge":"🚀","context":"128K","speed":"متوسط","tag":"بناء مشاريع","vision":False,"strength":"projects"},
    {"id":"qwen/qwen2.5-vl-72b-instruct:free","name":"Qwen2.5 VL 72B","desc":"يقرأ الصور • الأقوى vision مجاناً","badge":"👁️","context":"128K","speed":"متوسط","tag":"يرى الصور","vision":True,"strength":"vision"},
    {"id":"qwen/qwen2.5-vl-32b-instruct:free","name":"Qwen2.5 VL 32B","desc":"يقرأ الصور • سريع","badge":"🔍","context":"128K","speed":"سريع","tag":"يرى الصور","vision":True,"strength":"vision"},
    {"id":"google/gemma-4-31b-it:free","name":"Gemma 4 31B","desc":"يقرأ الصور • 256K context","badge":"🌟","context":"256K","speed":"متوسط","tag":"يرى الصور","vision":True,"strength":"vision"},
    {"id":"openai/gpt-oss-20b:free","name":"GPT-OSS 20B","desc":"سريع ودقيق","badge":"⚡","context":"128K","speed":"سريع","tag":"سريع","vision":False,"strength":"general"},
    {"id":"meta-llama/llama-4-scout:free","name":"Llama 4 Scout","desc":"الأسرع استجابةً","badge":"⚡","context":"128K","speed":"سريع جداً","tag":"الأسرع","vision":False,"strength":"general"},
]

VISION_IDS = [m["id"] for m in MODELS if m["vision"]]
CODING_IDS = [m["id"] for m in MODELS if not m["vision"]]

# ── Markdown → HTML ────────────────────────────────────────────────────────────
def markdown_to_html(text: str) -> str:
    code_blocks: list[tuple[str,str]] = []

    def extract(m: re.Match) -> str:
        lang = m.group(1) or "text"
        code = m.group(2)
        idx  = len(code_blocks)
        code_blocks.append((lang, code))
        return f"___CODE_{idx}___"

    text = re.sub(r"```(\w+)?\n(.*?)```", extract, text, flags=re.DOTALL)
    text = text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    text = re.sub(r"`([^`]+)`",        r"<code>\1</code>",         text)
    text = re.sub(r"^### (.*?)$",  r"<h3>\1</h3>",  text, flags=re.MULTILINE)
    text = re.sub(r"^## (.*?)$",   r"<h2>\1</h2>",  text, flags=re.MULTILINE)
    text = re.sub(r"^# (.*?)$",    r"<h1>\1</h1>",  text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.*?)\*",     r"<em>\1</em>",         text)
    text = re.sub(r"^> (.*?)$",    r"<blockquote>\1</blockquote>", text, flags=re.MULTILINE)
    text = re.sub(r"^\d+\.\s+(.*?)$", r"<li>\1</li>", text, flags=re.MULTILINE)
    text = re.sub(r"^[-*]\s+(.*?)$",  r"<li>\1</li>", text, flags=re.MULTILINE)

    result = []
    for p in text.split("\n\n"):
        p = p.strip()
        if p:
            p = p.replace("\n","<br>")
            if not p.startswith("<"):
                p = f"<p>{p}</p>"
            result.append(p)
    text = "\n".join(result)

    for idx,(lang,code) in enumerate(code_blocks):
        esc  = code.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        b64  = base64.b64encode(code.encode()).decode()
        html = (f'<div class="cw2"><div class="ch"><span>{lang or "code"}</span>'
                f'<button onclick="cpy(this,\'{b64}\')">📋 نسخ</button></div>'
                f'<pre><code>{esc}</code></pre></div>')
        text = text.replace(f"___CODE_{idx}___", html)
    return text

# ── File extraction for download ───────────────────────────────────────────────
_FPAT = re.compile(
    r"###\s*(\S+\.(?:py|html|css|js|json|txt|md|jsx|ts|tsx|vue|php|"
    r"java|cpp|c|go|rs|swift|kt|dart|rb|sh|sql|xml|yaml|yml))\s*\n*"
    r"```(?:\w+)?\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

def extract_files(text: str) -> dict[str,str]:
    files = {f.strip(): c.strip() for f,c in _FPAT.findall(text)}
    if not files:
        m = re.search(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
        if m:
            files[_guess(m.group(1))] = m.group(1).strip()
    return files

def _guess(c: str) -> str:
    cl = c.lower().strip()
    if cl.startswith(("<!doctype","<html")):                      return "index.html"
    if any(x in cl for x in ("fastapi","flask")) or cl.startswith("import "): return "app.py"
    if cl.startswith(("const ","let ")) or "function " in cl:    return "script.js"
    if "body {" in cl:                                            return "style.css"
    if cl.startswith(("{","[")):                                  return "data.json"
    return "code.txt"

# ── Read compressed files ──────────────────────────────────────────────────────
MAX_FC = 12_000

def read_file(fname: str, raw: bytes, ct: str) -> str:
    nl = fname.lower()
    if nl.endswith(".zip") or "zip" in ct:
        parts = [f"[ZIP: {fname}]"]
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                for i in zf.infolist():
                    if i.is_dir(): continue
                    try: parts.append(f"\n--- {i.filename} ---\n{zf.read(i.filename).decode('utf-8',errors='replace')[:3000]}")
                    except: parts.append(f"\n--- {i.filename} --- (binary)")
        except Exception as e: parts.append(f"(error: {e})")
        return "\n".join(parts)[:MAX_FC]
    if nl.endswith((".tar",".tar.gz",".tgz",".tar.bz2")) or "tar" in ct:
        parts = [f"[TAR: {fname}]"]
        try:
            with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
                for m in tf.getmembers():
                    if not m.isfile(): continue
                    try:
                        f = tf.extractfile(m)
                        if f: parts.append(f"\n--- {m.name} ---\n{f.read().decode('utf-8',errors='replace')[:3000]}")
                    except: parts.append(f"\n--- {m.name} --- (binary)")
        except Exception as e: parts.append(f"(error: {e})")
        return "\n".join(parts)[:MAX_FC]
    try:    return f"[{fname}]\n{raw.decode('utf-8',errors='replace')}"[:MAX_FC]
    except: return f"(cannot read: {fname})"

# ── Image URL (Pollinations) ───────────────────────────────────────────────────
def img_url(p: str, w=1024, h=1024, model="flux") -> str:
    return (f"https://image.pollinations.ai/prompt/{urllib.parse.quote(p)}"
            f"?width={w}&height={h}&model={model}&nologo=true&seed={abs(hash(p))%99999}")

# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html",
        {"request": request, "models": MODELS, "default_model": MODELS[0]["id"]})

@app.get("/models")
async def get_models(): return JSONResponse(MODELS)

@app.post("/stop")
async def stop():
    ev = stop_flags.get("default")
    if ev: ev.set()
    return JSONResponse({"ok": True})

# ── Download — serves RAW code text (fixes garbled file bug) ──────────────────
@app.post("/download")
async def download(filename: str = Form(...), code_b64: str = Form(...)):
    """
    code_b64: base64-encoded raw source code.
    This ensures the downloaded file is exact source, not HTML-escaped content.
    """
    try:
        code = base64.b64decode(code_b64).decode("utf-8")
    except Exception:
        return JSONResponse({"error": "invalid payload"}, status_code=400)

    ext = filename.rsplit(".",1)[-1].lower()
    mime = {
        "py":"text/x-python","js":"application/javascript","ts":"text/typescript",
        "html":"text/html","css":"text/css","json":"application/json",
        "txt":"text/plain","md":"text/markdown","sh":"text/x-sh",
        "sql":"text/x-sql","xml":"text/xml","yaml":"text/yaml","yml":"text/yaml",
        "jsx":"text/javascript","tsx":"text/typescript","vue":"text/javascript",
        "go":"text/x-go","rs":"text/x-rust","cpp":"text/x-c++src",
    }
    return Response(
        content=code,
        media_type=mime.get(ext,"text/plain"),
        headers={"Content-Disposition": f'attachment; filename="{filename}"',
                 "Content-Type": f'{mime.get(ext,"text/plain")}; charset=utf-8'},
    )

# ── Streaming chat ─────────────────────────────────────────────────────────────
@app.post("/chat-stream")
async def chat_stream(
    request:  Request,
    prompt:   str  = Form(...),
    model_id: str  = Form(None),
    # up to 25 files
    files: List[UploadFile] = File(default=[]),
):
    if not OPENROUTER_API_KEY:
        async def _e():
            yield f"data: {json.dumps({'error':'OPENROUTER_API_KEY not set'})}\n\n"
        return StreamingResponse(_e(), media_type="text/event-stream")

    valid_ids  = [m["id"] for m in MODELS]
    chosen     = model_id if model_id in valid_ids else MODELS[0]["id"]
    user_text  = prompt
    img_parts: list[dict] = []
    has_image  = False
    switched   = False

    # Process up to 25 attachments
    real_files = [f for f in (files or []) if f and f.filename][:25]
    for f in real_files:
        try:
            raw = await f.read()
            mt  = f.content_type or ""
            if mt.startswith("image/"):
                has_image = True
                b64 = base64.b64encode(raw).decode()
                img_parts.append({"type":"image_url","image_url":{"url":f"data:{mt};base64,{b64}"}})
            else:
                user_text += f"\n\n{read_file(f.filename, raw, mt)}"
        except Exception as exc:
            logger.error(f"File {f.filename}: {exc}")

    # Auto-switch to vision model if images present
    if has_image:
        obj = next((m for m in MODELS if m["id"]==chosen), None)
        if not (obj and obj["vision"]):
            chosen   = VISION_IDS[0] if VISION_IDS else chosen
            switched = True

    ordered = VISION_IDS if has_image else ([chosen]+[mid for mid in CODING_IDS if mid!=chosen])

    sid = "default"
    chat_sessions.setdefault(sid, [])
    session = chat_sessions[sid]

    stop_ev = asyncio.Event()
    stop_flags[sid] = stop_ev

    # Build message content
    content_parts: list = [{"type":"text","text":user_text}] + img_parts
    msg_content = content_parts if (img_parts or len(content_parts)>1) else user_text

    messages = [{"role":"system","content":SYSTEM_PROMPT}]
    messages.extend(session[-8:])
    messages.append({"role":"user","content":msg_content})

    hdrs = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://leo-ai.app",
        "X-Title":       "Leo-AI",
    }

    async def streamer():
        full_text  = ""
        used_model = ordered[0] if ordered else chosen
        last_err   = ""

        for mid in ordered:
            if stop_ev.is_set(): break
            logger.info(f"Trying: {mid}")
            try:
                to = httpx.Timeout(connect=8.0, read=90.0, write=10.0, pool=5.0)
                async with httpx.AsyncClient(timeout=to) as client:
                    async with client.stream("POST",
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=hdrs,
                        json={"model":mid,"messages":messages,"temperature":0.3,
                              "max_tokens":4000,"stream":True},
                    ) as resp:
                        if resp.status_code != 200:
                            body = await resp.aread()
                            try:    last_err = json.loads(body).get("error",{}).get("message","")
                            except: last_err = body.decode(errors="replace")[:200]
                            logger.warning(f"{mid} HTTP {resp.status_code}: {last_err}")
                            continue

                        used_model = mid
                        buf = ""
                        got_first = False

                        async for chunk in resp.aiter_text():
                            if stop_ev.is_set(): break
                            buf += chunk
                            lines = buf.split("\n"); buf = lines.pop()
                            for line in lines:
                                line = line.strip()
                                if not line.startswith("data:"): continue
                                ds = line[5:].strip()
                                if ds == "[DONE]": break
                                try:
                                    delta = json.loads(ds)["choices"][0]["delta"].get("content","")
                                    if delta:
                                        if not got_first:
                                            got_first = True
                                            yield f"data: {json.dumps({'first':True})}\n\n"
                                        full_text += delta
                                        yield f"data: {json.dumps({'token':delta})}\n\n"
                                except: continue
                break  # success

            except httpx.ReadTimeout:
                last_err = "timeout"; logger.warning(f"{mid} timeout"); continue
            except asyncio.CancelledError: break
            except Exception as exc:
                last_err = str(exc); logger.warning(f"{mid}: {exc}"); continue

        if stop_ev.is_set() and full_text:
            full_text += "\n\n*(⏹ توقف)*"
        if not full_text:
            full_text = f"⚠️ فشلت كل النماذج.\n\nالخطأ: `{last_err}`"
            yield f"data: {json.dumps({'token':full_text})}\n\n"

        session.append({"role":"user",      "content":prompt})
        session.append({"role":"assistant", "content":full_text})

        used_name   = next((m["name"] for m in MODELS if m["id"]==used_model), used_model.split("/")[-1])
        files_found = extract_files(full_text)

        # Build files payload with base64-encoded raw code (for correct download)
        files_b64 = {
            name: base64.b64encode(code.encode("utf-8")).decode()
            for name, code in files_found.items()
        }

        yield f"data: {json.dumps({'done':True,'html':markdown_to_html(full_text),'files':files_b64,'used_model':used_name,'switched':switched,'raw':full_text})}\n\n"

    return StreamingResponse(streamer(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

# ── Image generation ───────────────────────────────────────────────────────────
@app.post("/generate-image")
async def generate_image(prompt: str=Form(...), size: str=Form("1024x1024"), model: str=Form("flux")):
    try:
        w,h = map(int, size.split("x")) if "x" in size else (1024,1024)
        return JSONResponse({"url": img_url(prompt,w,h,model), "prompt": prompt})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/clear")
async def clear():
    chat_sessions["default"] = []
    ev = stop_flags.get("default")
    if ev: ev.set()
    return JSONResponse({"ok": True})

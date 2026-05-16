import os
import base64
import requests
import logging
import re
import zipfile
import tarfile
import io
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, Response, JSONResponse
from fastapi.templating import Jinja2Templates

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

chat_sessions = {}
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

# ─── قائمة الموديلات ────────────────────────────────────────────────────────
MODELS = [
    {
        "id":      "qwen/qwen3-coder-480b-a35b-instruct:free",
        "name":    "Qwen3 Coder 480B",
        "desc":    "الأقوى للكود حالياً",
        "badge":   "🥇",
        "context": "262K",
        "speed":   "متوسط",
        "tag":     "أفضل للكود",
    },
    {
        "id":      "deepseek/deepseek-r1:free",
        "name":    "DeepSeek R1",
        "desc":    "تفكير عميق ومنطق ممتاز",
        "badge":   "🧠",
        "context": "128K",
        "speed":   "بطيء",
        "tag":     "أفضل للتفكير",
    },
    {
        "id":      "openai/gpt-oss-20b:free",
        "name":    "GPT-OSS 20B",
        "desc":    "سريع ودقيق في الكود",
        "badge":   "⚡",
        "context": "128K",
        "speed":   "سريع",
        "tag":     "سريع",
    },
    {
        "id":      "meta-llama/llama-4-maverick:free",
        "name":    "Llama 4 Maverick",
        "desc":    "يدعم الصور والنصوص الطويلة",
        "badge":   "🦙",
        "context": "128K",
        "speed":   "متوسط",
        "tag":     "يدعم الصور",
    },
    {
        "id":      "meta-llama/llama-4-scout:free",
        "name":    "Llama 4 Scout",
        "desc":    "الأسرع استجابةً",
        "badge":   "🚀",
        "context": "128K",
        "speed":   "سريع جداً",
        "tag":     "الأسرع",
    },
    {
        "id":      "nvidia/nemotron-3-super-120b-a12b:free",
        "name":    "Nemotron 120B",
        "desc":    "نافذة سياق ضخمة 1M",
        "badge":   "🔬",
        "context": "1M",
        "speed":   "متوسط",
        "tag":     "سياق ضخم",
    },
]

SYSTEM_PROMPT = (
    "أنت مساعد برمجي متخصص. مهمتك الوحيدة هي مساعدة المستخدم في كتابة وتعديل وشرح الأكواد البرمجية. "
    "اكتب الكود دائماً داخل بلوكات Markdown. إذا كانت هناك عدة ملفات استخدم ### filename.ext قبل كل بلوك. "
    "كن دقيقاً ومختصراً. رد دائماً باللغة العربية."
)

# ─── Markdown → HTML ────────────────────────────────────────────────────────
def markdown_to_html(text: str) -> str:
    code_blocks = []

    def extract_code(match):
        lang = match.group(1) or "text"
        code = match.group(2)
        idx = len(code_blocks)
        code_blocks.append((lang, code))
        return f"___CODE_{idx}___"

    text = re.sub(r'```(\w+)?\n(.*?)```', extract_code, text, flags=re.DOTALL)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    text = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.*?)$',  r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.*?)$',   r'<h1>\1</h1>', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.*?)\*',     r'<em>\1</em>', text)
    text = re.sub(r'^> (.*?)$', r'<blockquote>\1</blockquote>', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\.\s+(.*?)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    text = re.sub(r'^[-*]\s+(.*?)$',  r'<li>\1</li>', text, flags=re.MULTILINE)

    paragraphs = text.split("\n\n")
    result = []
    for p in paragraphs:
        p = p.strip()
        if p:
            p = p.replace("\n", "<br>")
            if not p.startswith("<"):
                p = "<p>" + p + "</p>"
            result.append(p)
    text = "\n".join(result)

    for idx, (lang, code) in enumerate(code_blocks):
        escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        b64 = base64.b64encode(code.encode("utf-8")).decode("utf-8")
        display = lang if lang else "code"
        html = (
            '<div class="code-wrap">'
            '<div class="code-hdr">'
            f'<span>{display}</span>'
            f'<button onclick="cpy(this,\'{b64}\')">&#128203; نسخ</button>'
            '</div><pre><code>' + escaped + '</code></pre></div>'
        )
        text = text.replace(f"___CODE_{idx}___", html)
    return text

# ─── استخراج الملفات ────────────────────────────────────────────────────────
def extract_files(text: str) -> dict:
    files = {}
    pattern = (
        r'###\s*(\S+\.(?:py|html|css|js|json|txt|md|jsx|ts|tsx|vue|php|'
        r'java|cpp|c|go|rs|swift|kt|dart|rb|sh|sql|xml|yaml|yml))\s*\n*'
        r'```(?:\w+)?\n(.*?)```'
    )
    for fname, code in re.findall(pattern, text, re.DOTALL | re.IGNORECASE):
        files[fname.strip()] = code.strip()
    if not files:
        m = re.search(r'```(?:\w+)?\n(.*?)```', text, re.DOTALL)
        if m:
            c = m.group(1)
            files[guess_name(c)] = c.strip()
    return files

def guess_name(content: str) -> str:
    cl = content.lower().strip()
    if cl.startswith("<!doctype") or cl.startswith("<html"):   return "index.html"
    elif "fastapi" in cl or "flask" in cl or cl.startswith("import "): return "app.py"
    elif cl.startswith("const ") or cl.startswith("let ") or "function " in cl: return "script.js"
    elif "body {" in cl or ".class" in cl: return "style.css"
    elif cl.startswith("{") or cl.startswith("["): return "data.json"
    return "code.txt"

# ─── قراءة الملفات المضغوطة ─────────────────────────────────────────────────
MAX_FILE_TEXT = 12_000

def read_file_content(filename: str, raw_bytes: bytes, content_type: str) -> str:
    name_lower = filename.lower()

    if name_lower.endswith(".zip") or content_type in ("application/zip", "application/x-zip-compressed"):
        parts = [f"[محتوى الأرشيف: {filename}]"]
        try:
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                for info in zf.infolist():
                    if info.is_dir(): continue
                    try:
                        data = zf.read(info.filename)
                        parts.append(f"\n--- {info.filename} ---\n{data.decode('utf-8', errors='replace')[:4000]}")
                    except Exception:
                        parts.append(f"\n--- {info.filename} --- (ثنائي)")
        except Exception as e:
            parts.append(f"(فشل: {e})")
        return "\n".join(parts)[:MAX_FILE_TEXT]

    if name_lower.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2")) or "tar" in content_type:
        parts = [f"[محتوى الأرشيف: {filename}]"]
        try:
            with tarfile.open(fileobj=io.BytesIO(raw_bytes)) as tf:
                for member in tf.getmembers():
                    if not member.isfile(): continue
                    try:
                        f = tf.extractfile(member)
                        if f:
                            parts.append(f"\n--- {member.name} ---\n{f.read().decode('utf-8', errors='replace')[:4000]}")
                    except Exception:
                        parts.append(f"\n--- {member.name} --- (ثنائي)")
        except Exception as e:
            parts.append(f"(فشل: {e})")
        return "\n".join(parts)[:MAX_FILE_TEXT]

    try:
        return f"[ملف: {filename}]\n{raw_bytes.decode('utf-8', errors='replace')}"[:MAX_FILE_TEXT]
    except Exception:
        return f"(لم يمكن قراءة الملف: {filename})"

# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request":       request,
        "models":        MODELS,
        "default_model": MODELS[0]["id"],
    })


@app.get("/models", response_class=JSONResponse)
async def get_models():
    return JSONResponse(MODELS)


@app.post("/chat", response_class=JSONResponse)
async def chat(
    request: Request,
    prompt:   str = Form(...),
    model_id: str = Form(None),
    file: UploadFile = File(None),
):
    if not OPENROUTER_API_KEY:
        return JSONResponse({"error": "API key not set"}, status_code=500)

    # الموديل المختار + fallback
    valid_ids = [m["id"] for m in MODELS]
    chosen   = model_id if model_id in valid_ids else MODELS[0]["id"]
    ordered  = [chosen] + [mid for mid in valid_ids if mid != chosen]

    sid = "default"
    if sid not in chat_sessions:
        chat_sessions[sid] = []
    session = chat_sessions[sid]

    user_text     = prompt
    image_content = None

    if file and file.filename:
        try:
            raw = await file.read()
            mt  = file.content_type or ""
            if mt.startswith("image/"):
                b64 = base64.b64encode(raw).decode("utf-8")
                image_content = {"type": "image_url", "image_url": {"url": f"data:{mt};base64,{b64}"}}
            else:
                user_text = f"{prompt}\n\n{read_file_content(file.filename, raw, mt)}"
        except Exception as e:
            logger.error(f"File error: {e}")

    msg_content = [{"type": "text", "text": user_text}, image_content] if image_content else user_text

    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    msgs    = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs.extend(session[-8:])
    msgs.append({"role": "user", "content": msg_content})

    ai_resp    = ""
    used_model = chosen
    last_err   = ""

    for mid in ordered:
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json={"model": mid, "messages": msgs, "temperature": 0.5, "max_tokens": 3000},
                timeout=30,
            )
            j = r.json()
            if "choices" in j and j["choices"]:
                ai_resp    = j["choices"][0]["message"]["content"]
                used_model = mid
                break
            elif "error" in j:
                last_err = j["error"].get("message", "unknown")
        except Exception as exc:
            last_err = str(exc)

    if not ai_resp:
        ai_resp = f"عذراً، فشلت كل النماذج. الخطأ: {last_err}"

    session.append({"role": "user",      "content": prompt})
    session.append({"role": "assistant", "content": ai_resp})

    used_name = next((m["name"] for m in MODELS if m["id"] == used_model), used_model.split("/")[-1])

    return JSONResponse({
        "ai_html":    markdown_to_html(ai_resp),
        "files":      extract_files(ai_resp),
        "used_model": used_name,
    })


@app.post("/download")
async def download(filename: str = Form(...), code_content: str = Form(...)):
    ext = filename.split(".")[-1].lower()
    mime_map = {
        "py":"text/x-python","js":"application/javascript","html":"text/html",
        "css":"text/css","json":"application/json","txt":"text/plain",
        "md":"text/markdown","sh":"text/x-sh","sql":"text/x-sql",
        "xml":"text/xml","yaml":"text/yaml","yml":"text/yaml",
    }
    return Response(
        content=code_content,
        media_type=mime_map.get(ext, "text/plain"),
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.post("/clear")
async def clear():
    chat_sessions["default"] = []
    return JSONResponse({"ok": True})

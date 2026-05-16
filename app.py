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

# ✅ كل جلسة مستخدم محفوظة بشكل صحيح
chat_sessions = {}
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

FALLBACK_MODELS = [
    "google/gemma-3-27b-it:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-4-26b-a4b-it:free",
    "openai/gpt-oss-120b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
]

SYSTEM_PROMPT = (
    "أنت مساعد برمجي متخصص. مهمتك الوحيدة هي مساعدة المستخدم في كتابة وتعديل وشرح الأكواد البرمجية. "
    "اكتب الكود دائماً داخل بلوكات Markdown. إذا كانت هناك عدة ملفات استخدم ### filename.ext قبل كل بلوك. "
    "كن دقيقاً ومختصراً. رد دائماً باللغة العربية."
)

# ─── تحويل Markdown إلى HTML ───────────────────────────────────────────────
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

# ─── استخراج الملفات من الرد ────────────────────────────────────────────────
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
    if cl.startswith("<!doctype") or cl.startswith("<html"):
        return "index.html"
    elif "fastapi" in cl or "flask" in cl or cl.startswith("import "):
        return "app.py"
    elif cl.startswith("const ") or cl.startswith("let ") or "function " in cl:
        return "script.js"
    elif "body {" in cl or ".class" in cl:
        return "style.css"
    elif cl.startswith("{") or cl.startswith("["):
        return "data.json"
    return "code.txt"

# ─── قراءة الملفات المضغوطة والنصية ────────────────────────────────────────
MAX_FILE_TEXT = 12_000  # حد أقصى للنصوص المرسلة للنموذج (حرف)

def read_file_content(filename: str, raw_bytes: bytes, content_type: str) -> str:
    """يُعيد نص يُضاف لرسالة المستخدم."""
    name_lower = filename.lower()

    # ── ملفات ZIP ──────────────────────────────────────────────────────────
    if name_lower.endswith(".zip") or content_type in ("application/zip", "application/x-zip-compressed"):
        parts = [f"[محتوى الأرشيف: {filename}]"]
        try:
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    inner = info.filename
                    try:
                        data = zf.read(inner)
                        text = data.decode("utf-8", errors="replace")
                        parts.append(f"\n--- {inner} ---\n{text[:4000]}")
                    except Exception:
                        parts.append(f"\n--- {inner} --- (ثنائي، لا يمكن قراءته)")
        except Exception as e:
            parts.append(f"(فشل فتح الأرشيف: {e})")
        return "\n".join(parts)[:MAX_FILE_TEXT]

    # ── ملفات TAR / TAR.GZ / TGZ ───────────────────────────────────────────
    if (name_lower.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2")) or
            "tar" in content_type):
        parts = [f"[محتوى الأرشيف: {filename}]"]
        try:
            with tarfile.open(fileobj=io.BytesIO(raw_bytes)) as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    try:
                        f = tf.extractfile(member)
                        if f:
                            text = f.read().decode("utf-8", errors="replace")
                            parts.append(f"\n--- {member.name} ---\n{text[:4000]}")
                    except Exception:
                        parts.append(f"\n--- {member.name} --- (ثنائي)")
        except Exception as e:
            parts.append(f"(فشل فتح الأرشيف: {e})")
        return "\n".join(parts)[:MAX_FILE_TEXT]

    # ── نصوص عادية ─────────────────────────────────────────────────────────
    try:
        text = raw_bytes.decode("utf-8", errors="replace")
        return f"[ملف: {filename}]\n{text}"[:MAX_FILE_TEXT]
    except Exception:
        return f"(لم يمكن قراءة الملف: {filename})"

# ─── المسارات ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # ✅ الصفحة تبدأ فاضية دائماً
    return templates.TemplateResponse("index.html", {
        "request": request,
        "chat_history": [],
        "files": {},
        "user_prompt": ""
    })


@app.post("/chat", response_class=JSONResponse)
async def chat(
    request: Request,
    prompt: str = Form(...),
    file: UploadFile = File(None),
):
    if not OPENROUTER_API_KEY:
        return JSONResponse({"error": "API key not set"}, status_code=500)

    sid = "default"
    if sid not in chat_sessions:
        chat_sessions[sid] = []
    session = chat_sessions[sid]

    # ── بناء محتوى رسالة المستخدم ──────────────────────────────────────────
    user_text = prompt
    image_content = None

    if file and file.filename:
        try:
            raw = await file.read()
            mt = file.content_type or ""
            if mt.startswith("image/"):
                b64 = base64.b64encode(raw).decode("utf-8")
                image_content = {"type": "image_url",
                                 "image_url": {"url": f"data:{mt};base64,{b64}"}}
            else:
                extra = read_file_content(file.filename, raw, mt)
                user_text = f"{prompt}\n\n{extra}"
        except Exception as e:
            logger.error(f"File read error: {e}")

    if image_content:
        msg_content = [{"type": "text", "text": user_text}, image_content]
    else:
        msg_content = user_text

    # ── استدعاء النموذج ─────────────────────────────────────────────────────
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs.extend(session[-8:])   # آخر 4 تبادلات
    msgs.append({"role": "user", "content": msg_content})

    ai_resp = ""
    last_err = ""
    for model in FALLBACK_MODELS:
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json={"model": model, "messages": msgs,
                      "temperature": 0.5, "max_tokens": 3000},
                timeout=25,
            )
            j = r.json()
            if "choices" in j and j["choices"]:
                ai_resp = j["choices"][0]["message"]["content"]
                break
            elif "error" in j:
                last_err = j["error"].get("message", "unknown error")
        except Exception as exc:
            last_err = str(exc)
            continue

    if not ai_resp:
        ai_resp = f"عذراً، فشلت كل النماذج. الخطأ: {last_err}"

    # ── حفظ الجلسة ──────────────────────────────────────────────────────────
    session.append({"role": "user",      "content": prompt})   # نص فقط للجلسة
    session.append({"role": "assistant", "content": ai_resp})

    files = extract_files(ai_resp)

    return JSONResponse({
        "user":    prompt,
        "ai_html": markdown_to_html(ai_resp),
        "files":   files,
    })


@app.post("/download")
async def download(filename: str = Form(...), code_content: str = Form(...)):
    ext = filename.split(".")[-1].lower()
    mime_map = {
        "py": "text/x-python", "js": "application/javascript",
        "html": "text/html",   "css": "text/css",
        "json": "application/json", "txt": "text/plain",
        "md": "text/markdown", "sh": "text/x-sh",
        "sql": "text/x-sql",   "xml": "text/xml",
        "yaml": "text/yaml",   "yml": "text/yaml",
    }
    return Response(
        content=code_content,
        media_type=mime_map.get(ext, "text/plain"),
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.post("/clear")
async def clear():
    # ✅ مسح فوري وصحيح
    chat_sessions["default"] = []
    return JSONResponse({"ok": True})

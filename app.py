import os
import base64
import requests
import logging
import re
import zipfile
import tarfile
import io
import json
import urllib.parse
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, Response, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

chat_sessions = {}
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

# ─── الموديلات ────────────────────────────────────────────────────────────────
MODELS = [
    {
        "id":      "qwen/qwen3-coder-480b-a35b-instruct:free",
        "name":    "Qwen3 Coder 480B",
        "desc":    "الأقوى للكود • 262K context",
        "badge":   "🥇",
        "context": "262K",
        "speed":   "متوسط",
        "tag":     "أفضل للكود",
        "vision":  False,
    },
    {
        "id":      "qwen/qwen2.5-vl-72b-instruct:free",
        "name":    "Qwen2.5 VL 72B",
        "desc":    "يقرأ الصور • الأقوى في vision مجاناً",
        "badge":   "👁️",
        "context": "128K",
        "speed":   "متوسط",
        "tag":     "يرى الصور",
        "vision":  True,
    },
    {
        "id":      "qwen/qwen2.5-vl-32b-instruct:free",
        "name":    "Qwen2.5 VL 32B",
        "desc":    "يقرأ الصور • سريع وخفيف",
        "badge":   "🔍",
        "context": "128K",
        "speed":   "سريع",
        "tag":     "يرى الصور",
        "vision":  True,
    },
    {
        "id":      "google/gemma-4-31b-it:free",
        "name":    "Gemma 4 31B",
        "desc":    "يقرأ الصور • 256K context • Google",
        "badge":   "🌟",
        "context": "256K",
        "speed":   "متوسط",
        "tag":     "يرى الصور",
        "vision":  True,
    },
    {
        "id":      "nvidia/nemotron-nano-12b-v2-vl:free",
        "name":    "Nemotron VL 12B",
        "desc":    "يقرأ الصور • 300K context",
        "badge":   "🔬",
        "context": "300K",
        "speed":   "سريع",
        "tag":     "يرى الصور",
        "vision":  True,
    },
    {
        "id":      "deepseek/deepseek-r1:free",
        "name":    "DeepSeek R1",
        "desc":    "تفكير عميق ومنطق ممتاز",
        "badge":   "🧠",
        "context": "128K",
        "speed":   "بطيء",
        "tag":     "أفضل للتفكير",
        "vision":  False,
    },
    {
        "id":      "openai/gpt-oss-20b:free",
        "name":    "GPT-OSS 20B",
        "desc":    "سريع ودقيق في الكود",
        "badge":   "⚡",
        "context": "128K",
        "speed":   "سريع",
        "tag":     "سريع",
        "vision":  False,
    },
    {
        "id":      "meta-llama/llama-4-scout:free",
        "name":    "Llama 4 Scout",
        "desc":    "الأسرع استجابةً",
        "badge":   "🚀",
        "context": "128K",
        "speed":   "سريع جداً",
        "tag":     "الأسرع",
        "vision":  False,
    },
]

VISION_IDS = [m["id"] for m in MODELS if m["vision"]]

SYSTEM_PROMPT = (
    "أنت مساعد برمجي متخصص. مهمتك مساعدة المستخدم في كتابة وتعديل وشرح الأكواد البرمجية "
    "وتحليل الصور المتعلقة بالبرمجة. "
    "اكتب الكود دائماً داخل بلوكات Markdown. إذا كانت هناك عدة ملفات استخدم ### filename.ext قبل كل بلوك. "
    "كن دقيقاً ومختصراً. رد دائماً باللغة العربية."
)

# ─── Markdown → HTML ─────────────────────────────────────────────────────────
def markdown_to_html(text: str) -> str:
    code_blocks = []

    def extract_code(match):
        lang = match.group(1) or "text"
        code = match.group(2)
        idx  = len(code_blocks)
        code_blocks.append((lang, code))
        return f"___CODE_{idx}___"

    text = re.sub(r'```(\w+)?\n(.*?)```', extract_code, text, flags=re.DOTALL)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r'`([^`]+)`',       r'<code>\1</code>', text)
    text = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.*?)$',  r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.*?)$',   r'<h1>\1</h1>', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.*?)\*',     r'<em>\1</em>',         text)
    text = re.sub(r'^> (.*?)$',   r'<blockquote>\1</blockquote>', text, flags=re.MULTILINE)
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
        escaped = code.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        b64     = base64.b64encode(code.encode("utf-8")).decode("utf-8")
        display = lang if lang else "code"
        html = (
            '<div class="cw2"><div class="ch">'
            f'<span>{display}</span>'
            f'<button onclick="cpy(this,\'{b64}\')">&#128203; نسخ</button>'
            '</div><pre><code>' + escaped + '</code></pre></div>'
        )
        text = text.replace(f"___CODE_{idx}___", html)
    return text

# ─── استخراج الملفات ──────────────────────────────────────────────────────────
def extract_files(text: str) -> dict:
    files   = {}
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
    if cl.startswith("<!doctype") or cl.startswith("<html"):            return "index.html"
    elif "fastapi" in cl or "flask" in cl or cl.startswith("import "): return "app.py"
    elif cl.startswith("const ") or "function " in cl:                  return "script.js"
    elif "body {" in cl or ".class" in cl:                              return "style.css"
    elif cl.startswith("{") or cl.startswith("["):                      return "data.json"
    return "code.txt"

# ─── قراءة الملفات المضغوطة ───────────────────────────────────────────────────
MAX_FILE_TEXT = 12_000

def read_file_content(filename: str, raw_bytes: bytes, content_type: str) -> str:
    name_lower = filename.lower()

    if name_lower.endswith(".zip") or "zip" in content_type:
        parts = [f"[أرشيف: {filename}]"]
        try:
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                for info in zf.infolist():
                    if info.is_dir(): continue
                    try:
                        data = zf.read(info.filename)
                        parts.append(f"\n--- {info.filename} ---\n{data.decode('utf-8',errors='replace')[:4000]}")
                    except Exception:
                        parts.append(f"\n--- {info.filename} --- (ثنائي)")
        except Exception as e:
            parts.append(f"(فشل: {e})")
        return "\n".join(parts)[:MAX_FILE_TEXT]

    if name_lower.endswith((".tar",".tar.gz",".tgz",".tar.bz2")) or "tar" in content_type:
        parts = [f"[أرشيف: {filename}]"]
        try:
            with tarfile.open(fileobj=io.BytesIO(raw_bytes)) as tf:
                for member in tf.getmembers():
                    if not member.isfile(): continue
                    try:
                        f = tf.extractfile(member)
                        if f:
                            parts.append(f"\n--- {member.name} ---\n{f.read().decode('utf-8',errors='replace')[:4000]}")
                    except Exception:
                        parts.append(f"\n--- {member.name} --- (ثنائي)")
        except Exception as e:
            parts.append(f"(فشل: {e})")
        return "\n".join(parts)[:MAX_FILE_TEXT]

    try:
        return f"[ملف: {filename}]\n{raw_bytes.decode('utf-8',errors='replace')}"[:MAX_FILE_TEXT]
    except Exception:
        return f"(لم يمكن قراءة الملف: {filename})"

# ─── توليد الصور ─────────────────────────────────────────────────────────────
def build_image_url(prompt: str, width: int = 1024, height: int = 1024, model: str = "flux") -> str:
    encoded = urllib.parse.quote(prompt)
    seed    = abs(hash(prompt)) % 99999
    return f"https://image.pollinations.ai/prompt/{encoded}?width={width}&height={height}&model={model}&nologo=true&seed={seed}"

# ─── Routes ───────────────────────────────────────────────────────────────────

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


# ─── STREAMING CHAT ────────────────────────────────────────────────────────────
@app.post("/chat-stream")
async def chat_stream(
    request:  Request,
    prompt:   str        = Form(...),
    model_id: str        = Form(None),
    file:     UploadFile = File(None),
):
    if not OPENROUTER_API_KEY:
        async def err():
            yield f"data: {json.dumps({'error': 'API key not set'})}\n\n"
        return StreamingResponse(err(), media_type="text/event-stream")

    # ── اختيار الموديل ────────────────────────────────────────────────────────
    valid_ids  = [m["id"] for m in MODELS]
    chosen     = model_id if model_id in valid_ids else MODELS[0]["id"]
    has_image  = False
    user_text  = prompt
    image_part = None
    switched   = False

    if file and file.filename:
        try:
            raw = await file.read()
            mt  = file.content_type or ""
            if mt.startswith("image/"):
                has_image  = True
                b64        = base64.b64encode(raw).decode("utf-8")
                image_part = {"type": "image_url", "image_url": {"url": f"data:{mt};base64,{b64}"}}
            else:
                user_text = f"{prompt}\n\n{read_file_content(file.filename, raw, mt)}"
        except Exception as e:
            logger.error(f"File error: {e}")

    # تحويل تلقائي لموديل vision إذا كانت صورة
    if has_image:
        obj = next((m for m in MODELS if m["id"] == chosen), None)
        if not (obj and obj["vision"]):
            chosen   = VISION_IDS[0]
            switched = True

    # ترتيب المحاولات
    if has_image:
        ordered = VISION_IDS
    else:
        ordered = [chosen] + [mid for mid in valid_ids if mid != chosen]

    # ── الجلسة ────────────────────────────────────────────────────────────────
    sid = "default"
    if sid not in chat_sessions:
        chat_sessions[sid] = []
    session = chat_sessions[sid]

    msg_content = [{"type": "text", "text": user_text}, image_part] if image_part else user_text

    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    msgs    = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs.extend(session[-8:])
    msgs.append({"role": "user", "content": msg_content})

    # ── دالة streaming ─────────────────────────────────────────────────────────
    async def streamer():
        full_text  = ""
        used_model = ordered[0]
        last_err   = ""
        succeeded  = False

        for mid in ordered:
            try:
                resp = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json={
                        "model":       mid,
                        "messages":    msgs,
                        "temperature": 0.5,
                        "max_tokens":  3000,
                        "stream":      True,        # ← streaming مفعّل
                    },
                    stream=True,
                    timeout=40,
                )

                # فحص أولي للخطأ
                if resp.status_code != 200:
                    try:
                        err_json = resp.json()
                        last_err = err_json.get("error", {}).get("message", str(resp.status_code))
                    except Exception:
                        last_err = str(resp.status_code)
                    continue

                used_model = mid

                # ── قراءة chunks ──────────────────────────────────────────────
                for line in resp.iter_lines():
                    if not line:
                        continue
                    line = line.decode("utf-8") if isinstance(line, bytes) else line
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk["choices"][0]["delta"].get("content", "")
                            if delta:
                                full_text += delta
                                # أرسل الـ token للمتصفح
                                yield f"data: {json.dumps({'token': delta})}\n\n"
                        except Exception:
                            continue

                succeeded = True
                break

            except Exception as exc:
                last_err = str(exc)
                logger.warning(f"Stream model {mid} failed: {exc}")
                continue

        if not succeeded or not full_text:
            full_text = f"عذراً، فشلت كل النماذج. الخطأ: {last_err}"
            yield f"data: {json.dumps({'token': full_text})}\n\n"

        # ── حفظ الجلسة وإرسال النهاية ─────────────────────────────────────────
        session.append({"role": "user",      "content": prompt})
        session.append({"role": "assistant", "content": full_text})

        used_name = next((m["name"] for m in MODELS if m["id"] == used_model), used_model.split("/")[-1])
        files_found = extract_files(full_text)
        final_html  = markdown_to_html(full_text)

        yield f"data: {json.dumps({'done': True, 'html': final_html, 'files': files_found, 'used_model': used_name, 'switched': switched})}\n\n"

    return StreamingResponse(
        streamer(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":  "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/generate-image", response_class=JSONResponse)
async def generate_image(prompt: str = Form(...), size: str = Form("1024x1024"), model: str = Form("flux")):
    try:
        w, h = (int(x) for x in size.split("x")) if "x" in size else (1024, 1024)
        url  = build_image_url(prompt, w, h, model)
        return JSONResponse({"url": url, "prompt": prompt})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


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
        media_type=mime_map.get(ext,"text/plain"),
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.post("/clear")
async def clear():
    chat_sessions["default"] = []
    return JSONResponse({"ok": True})

"""
Leo-AI v7.1 — Dual Account System with SQLite Persistence
- Admin: password login → full access (GitHub, project memory, all features)
- Guest: no password → limited access (chat only, no GitHub, saves locally)
- All chats persist in SQLite
- Auto title generation
- Delete/rename chats
- Save confirmation before leaving
- Working Preview
"""

import os, base64, logging, re, zipfile, io, json, sqlite3, uuid, hashlib, time, secrets
import urllib.parse, asyncio
from datetime import datetime
from contextlib import contextmanager

import httpx
from fastapi import FastAPI, Request, Form, UploadFile, File, Cookie
from fastapi.responses import (HTMLResponse, Response, JSONResponse,
                                StreamingResponse, RedirectResponse)
from fastapi.templating import Jinja2Templates
from typing import List

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("leo-ai")

app = FastAPI(title="Leo-AI", version="7.1.0")
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
DB_PATH = os.path.join(BASE_DIR, "leo_ai.db")

os.makedirs(TEMPLATES_DIR, exist_ok=True)
templates = Jinja2Templates(directory=TEMPLATES_DIR)

OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY")
GITHUB_TOKEN:       str | None = os.getenv("GITHUB_TOKEN")
APP_PASSWORD:       str        = os.getenv("APP_PASSWORD", "")

# ── Database Setup ───────────────────────────────────────────────────────────
def init_db():
    with get_db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            role TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            title TEXT,
            role TEXT NOT NULL,
            user_token TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            model TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS project_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_token TEXT NOT NULL,
            filename TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS generated_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id);
        CREATE INDEX IF NOT EXISTS idx_chats_user ON chats(user_token);
        """)

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

init_db()

# ── In-memory caches (synced with DB) ────────────────────────────────────────
stop_flags:      dict[str, asyncio.Event] = {}
shared_chats:    dict[str, dict] = {}

# ── Auth helpers ───────────────────────────────────────────────────────────────
def get_role(token: str | None) -> str:
    if not token:
        return "none"
    with get_db() as conn:
        row = conn.execute("SELECT role FROM sessions WHERE token = ?", (token,)).fetchone()
        if row:
            return row["role"]
    return "none"

def is_authed(token: str | None) -> bool:
    return get_role(token) in ("admin", "guest")

def is_admin(token: str | None) -> bool:
    return get_role(token) == "admin"

def check_auth(token: str | None):
    if not is_authed(token):
        return RedirectResponse(url="/login", status_code=302)
    return None

def check_admin(token: str | None):
    if not is_admin(token):
        return JSONResponse({"error": "هذه الميزة للمستخدم الرئيسي فقط"}, status_code=403)
    return None

def gh_headers() -> dict:
    h = {"Accept": "application/vnd.github.v3+json", "User-Agent": "Leo-AI/7.1"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"token {GITHUB_TOKEN}"
    return h

# ── Chat helpers ─────────────────────────────────────────────────────────────
def generate_title(text: str) -> str:
    """Generate a title from the first user message."""
    text = text.strip()
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"[#*`\[\]_]", "", text)
    text = text.strip()
    if not text:
        return "محادثة جديدة"
    title = text[:30].strip()
    if len(text) > 30:
        title += "..."
    return title

def get_chat_messages(chat_id: str) -> list:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT role, content, model FROM messages WHERE chat_id = ? ORDER BY created_at",
            (chat_id,)
        ).fetchall()
        return [{"role": r["role"], "content": r["content"], "model": r["model"]} for r in rows]

def get_user_chats(user_token: str) -> list:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at FROM chats WHERE user_token = ? ORDER BY updated_at DESC",
            (user_token,)
        ).fetchall()
        return [{"id": r["id"], "title": r["title"] or "بدون عنوان",
                 "created_at": r["created_at"], "updated_at": r["updated_at"]} for r in rows]

# ── System prompts ────────────────────────────────────────────────────────────
SYSTEM_PROMPT_ADMIN = """You are a senior software engineer, system architect, debugging specialist, security reviewer, and production-grade AI coding agent.

CORE ENGINEERING RULES:
- Think and analyze before writing code. Prioritize correctness, security, maintainability.
- Always identify root cause before fixing bugs. Never patch symptoms.
- For complex tasks: show implementation plan first, then code.
- Proactively flag security issues, architectural risks, and regressions.
- Write clean, modular, production-grade code with proper error handling.
- After every code block, add "⚠️ ملاحظات" section if there are important notes.
- When project files are provided, study them and respect existing architecture.
- Verify logic before claiming it works. Think about edge cases.

CODE REVIEW RULES:
- Analyze EVERY file systematically
- Report: 🐛 Bugs, 🔐 Security vulnerabilities, ⚡ Performance issues, 🏗️ Architecture problems, 📝 Code quality
- Provide specific line references when possible
- Show the fixed code for every issue found
- Rate overall code quality: /10
- Prioritize: CRITICAL > HIGH > MEDIUM > LOW

FORMAT:
- Detect user language and respond in SAME language
- Write code in Markdown code blocks with language tags
- For multi-file projects: ### filename.ext before each block
"""

SYSTEM_PROMPT_GUEST = """You are a helpful AI coding assistant. Help the user write, debug, and understand code.
- Be clear, concise, and helpful
- Write code in Markdown code blocks with language tags
- Detect user language and respond in SAME language
- Provide examples when helpful
"""

# ── Models ────────────────────────────────────────────────────────────────────
MODELS: list[dict] = [
    {"id":"deepseek/deepseek-chat-v3-0324:free","name":"DeepSeek Chat V3","desc":"أفضل للمحادثة العامة • 64K","badge":"💬","context":"64K","speed":"سريع","tag":"محادثة","vision":False,"strength":"general"},
    {"id":"deepseek/deepseek-r1:free","name":"DeepSeek R1","desc":"تفكير عميق • debugging","badge":"🧠","context":"128K","speed":"بطيء","tag":"تفكير عميق","vision":False,"strength":"reasoning"},
    {"id":"meta-llama/llama-4-maverick:free","name":"Llama 4 Maverick","desc":"أداء قوي عام","badge":"🦙","context":"128K","speed":"متوسط","tag":"عام","vision":False,"strength":"general"},
    {"id":"qwen/qwen3-235b-a22b:free","name":"Qwen3 235B","desc":"قوي للكود والمنطق","badge":"🥇","context":"128K","speed":"متوسط","tag":"كود + منطق","vision":False,"strength":"code"},
    {"id":"openai/gpt-oss-120b:free","name":"GPT-OSS 120B","desc":"نموذج مفتوح من OpenAI","badge":"⚡","context":"128K","speed":"متوسط","tag":"عام","vision":False,"strength":"general"},
    {"id":"nvidia/nemotron-nano-12b-v2-vl:free","name":"Nemotron Nano 2 VL","desc":"يرى الصور والفيديو","badge":"👁️","context":"128K","speed":"سريع","tag":"يرى الصور","vision":True,"strength":"vision"},
    {"id":"openrouter/owl-alpha:free","name":"Owl Alpha","desc":"وكيل ذكي • أدوات","badge":"🦉","context":"1M","speed":"متوسط","tag":"وكيل","vision":False,"strength":"agent"},
    {"id":"openrouter/pareto-code:free","name":"Pareto Code","desc":"أفضل للبرمجة تلقائياً","badge":"💻","context":"2M","speed":"سريع","tag":"كود تلقائي","vision":False,"strength":"code"},
]
VISION_IDS = [m["id"] for m in MODELS if m["vision"]]
CODING_IDS = [m["id"] for m in MODELS if not m["vision"]]

CODE_EXT = {".py",".js",".ts",".tsx",".jsx",".html",".css",".json",".md",".txt",
            ".yml",".yaml",".sh",".bash",".go",".rs",".java",".cpp",".c",".h",
            ".cs",".rb",".php",".vue",".sql",".toml",".env.example",".gitignore",
            ".dockerfile",".tf",".kt",".swift"}
SKIP_DIRS = {"node_modules",".git","dist","build","__pycache__",".venv","venv",
             "env",".next",".nuxt","coverage",".pytest_cache",".mypy_cache","target","vendor"}
MAX_FC = 15_000

# ── Markdown → HTML ───────────────────────────────────────────────────────────
def markdown_to_html(text: str) -> str:
    code_blocks: list[tuple[str,str]] = []
    def extract(m):
        lang=m.group(1) or "text"; code=m.group(2); idx=len(code_blocks)
        code_blocks.append((lang,code)); return f"___CODE_{idx}___"
    text=re.sub(r"```(\w+)?\n(.*?)```",extract,text,flags=re.DOTALL)
    text=text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    text=re.sub(r"`([^`]+)`",r"<code>\1</code>",text)
    text=re.sub(r"^### (.*?)$",r"<h3>\1</h3>",text,flags=re.MULTILINE)
    text=re.sub(r"^## (.*?)$", r"<h2>\1</h2>",text,flags=re.MULTILINE)
    text=re.sub(r"^# (.*?)$",  r"<h1>\1</h1>",text,flags=re.MULTILINE)
    text=re.sub(r"\*\*(.*?)\*\*",r"<strong>\1</strong>",text)
    text=re.sub(r"\*(.*?)\*",    r"<em>\1</em>",text)
    text=re.sub(r"^> (.*?)$",r"<blockquote>\1</blockquote>",text,flags=re.MULTILINE)
    text=re.sub(r"^\d+\.\s+(.*?)$",r"<li>\1</li>",text,flags=re.MULTILINE)
    text=re.sub(r"^[-*]\s+(.*?)$", r"<li>\1</li>",text,flags=re.MULTILINE)
    result=[]
    for p in text.split("\n\n"):
        p=p.strip()
        if p:
            p=p.replace("\n","<br>")
            if not p.startswith("<"): p=f"<p>{p}</p>"
            result.append(p)
    text="\n".join(result)
    for idx,(lang,code) in enumerate(code_blocks):
        esc_code=code.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        b64=base64.b64encode(code.encode()).decode()
        html=(f'<div class="cw2"><div class="ch"><span>{lang or "code"}</span>'
              f'<div class="ch-btns"><button onclick="cpy(this,\'{b64}\')">📋 نسخ</button>'
              f'<button onclick="explainCode(this,\'{b64}\')">💡 شرح</button></div></div>'
              f'<pre><code class="lang-{lang}">{esc_code}</code></pre></div>')
        text=text.replace(f"___CODE_{idx}___",html)
    return text

def extract_files(text):
    pat=re.compile(r"###\s*(\S+\.(?:py|html|css|js|json|txt|md|jsx|ts|tsx|vue|php|java|cpp|c|go|rs|rb|sh|sql|xml|yaml|yml))\s*\n*```(?:\w+)?\n(.*?)```",re.DOTALL|re.IGNORECASE)
    files={f.strip():c.strip() for f,c in pat.findall(text)}
    if not files:
        m=re.search(r"```(?:\w+)?\n(.*?)```",text,re.DOTALL)
        if m:
            c=m.group(1)
            cl=c.lower().strip()
            name="index.html" if cl.startswith(("<!doctype","<html")) else "app.py" if "fastapi" in cl or cl.startswith("import ") else "script.js" if "function " in cl else "data.json" if cl.startswith(("{","[")) else "code.txt"
            files[name]=c.strip()
    return files

def read_file(fname,raw,ct):
    nl=fname.lower()
    if nl.endswith(".pdf"): return _read_pdf(raw,fname)
    if nl.endswith(".zip") or "zip" in ct:
        parts=[f"[ZIP: {fname}]"]
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                for i in zf.infolist():
                    if i.is_dir() or i.file_size>500_000: continue
                    try: parts.append(f"\n--- {i.filename} ---\n{zf.read(i.filename).decode('utf-8',errors='replace')[:3000]}")
                    except: parts.append(f"\n--- {i.filename} --- (binary)")
        except Exception as e: parts.append(f"(error: {e})")
        return "\n".join(parts)[:MAX_FC]
    try: return f"[{fname}]\n{raw.decode('utf-8',errors='replace')}"[:MAX_FC]
    except: return f"(cannot read: {fname})"

def _read_pdf(raw,fname):
    if not HAS_PYPDF: return f"[PDF: {fname}] — pypdf not installed"
    try:
        reader=PdfReader(io.BytesIO(raw)); pages=[]
        for i,page in enumerate(reader.pages[:50]):
            t=page.extract_text() or ""
            if t.strip(): pages.append(f"[صفحة {i+1}]\n{t.strip()}")
        return f"[PDF: {fname}]\n\n"+"\n\n".join(pages)[:MAX_FC]
    except Exception as e: return f"[PDF: {fname}] error: {e}"

def build_project_context(user_token: str) -> str:
    with get_db() as conn:
        rows = conn.execute("SELECT filename, content FROM project_files WHERE user_token = ?", (user_token,)).fetchall()
        if not rows:
            return ""
        parts=["=== PROJECT CONTEXT ==="]
        for r in rows:
            parts.append(f"\n### {r['filename']}\n{r['content'][:3000]}")
        parts.append("=== END PROJECT CONTEXT ===\n")
        return "\n".join(parts)

def img_url(p,w=1024,h=1024,model="flux"):
    return (f"https://image.pollinations.ai/prompt/{urllib.parse.quote(p)}"
            f"?width={w}&height={h}&model={model}&nologo=true&seed={abs(hash(p))%99999}")

# ══════════════════════════════════════════════════════════════════════════════
# AUTH ROUTES
# ══════════════════════════════════════════════════════════════════════════════

LOGIN_PAGE = """<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<meta name="theme-color" content="#171717">
<title>Leo-AI</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#212121;color:#ececec;
     display:flex;align-items:center;justify-content:center;min-height:100vh;padding:20px}}
.wrap{{width:100%;max-width:340px;display:flex;flex-direction:column;gap:14px}}
.logo-wrap{{text-align:center;margin-bottom:8px}}
.logo{{width:70px;height:70px;background:linear-gradient(135deg,#19c37d,#ab68ff);
       border-radius:50%;display:flex;align-items:center;justify-content:center;
       font-size:34px;margin:0 auto 12px}}
h1{{font-size:22px;font-weight:700;text-align:center}}
.sub{{color:#8e8ea0;font-size:13px;text-align:center;margin-bottom:4px}}

/* Cards */
.card{{background:#2f2f2f;border:1px solid #404040;border-radius:14px;padding:20px}}
.card-title{{font-size:13px;font-weight:700;margin-bottom:12px;display:flex;align-items:center;gap:7px}}
.card-title .badge{{background:rgba(25,195,125,.15);color:#19c37d;border-radius:5px;padding:2px 7px;font-size:10px}}
.card-title .badge.g{{background:rgba(171,104,255,.15);color:#ab68ff}}

input[type=password]{{width:100%;background:#1a1a1a;color:#ececec;border:1px solid #404040;
      border-radius:9px;padding:12px 13px;font-size:15px;outline:none;
      text-align:center;letter-spacing:3px;margin-bottom:10px;font-family:inherit;
      transition:border-color .15s}}
input[type=password]:focus{{border-color:#19c37d}}
.btn{{width:100%;padding:12px;border:none;border-radius:9px;font-size:14px;
      font-weight:700;cursor:pointer;transition:opacity .15s}}
.btn:active{{opacity:.85}}
.btn-admin{{background:#19c37d;color:#111}}
.btn-guest{{background:#2f2f2f;color:#ececec;border:1px solid #404040}}
.btn-guest:active{{background:#3a3a3a}}
.err{{color:#ef4444;font-size:12px;text-align:center;min-height:16px;margin-top:2px}}
.divider{{display:flex;align-items:center;gap:10px;color:#555;font-size:11px;margin:4px 0}}
.divider::before,.divider::after{{content:'';flex:1;height:1px;background:#404040}}
.features{{display:flex;flex-direction:column;gap:5px;margin-top:10px}}
.feat{{display:flex;align-items:center;gap:7px;font-size:11.5px;color:#8e8ea0}}
.feat-dot{{width:6px;height:6px;border-radius:50%;flex-shrink:0}}
.feat-dot.ac{{background:#19c37d}}
.feat-dot.off{{background:#404040}}
</style>
</head>
<body>
<div class="wrap">
  <div class="logo-wrap">
    <div class="logo">🚀</div>
    <h1>Leo-AI</h1>
    <p class="sub">مساعد البرمجة الاحترافي</p>
  </div>

  <!-- Admin card -->
  <div class="card">
    <div class="card-title">🔐 المستخدم الرئيسي <span class="badge">وصول كامل</span></div>
    <form method="post" action="/login/admin">
      <input type="password" name="password" placeholder="كلمة المرور" autocomplete="current-password">
      <button class="btn btn-admin" type="submit">دخول ←</button>
    </form>
    <div class="err" id="err-admin">{admin_err}</div>
    <div class="features">
      <div class="feat"><span class="feat-dot ac"></span>GitHub browser + code review</div>
      <div class="feat"><span class="feat-dot ac"></span>ذاكرة المشروع الدائمة</div>
      <div class="feat"><span class="feat-dot ac"></span>جميع الميزات</div>
    </div>
  </div>

  <div class="divider">أو</div>

  <!-- Guest card -->
  <div class="card">
    <div class="card-title">👤 ضيف <span class="badge g">وصول محدود</span></div>
    <p style="font-size:12px;color:#8e8ea0;margin-bottom:12px">بدون كلمة مرور — محادثة مباشرة</p>
    <form method="post" action="/login/guest">
      <button class="btn btn-guest" type="submit">دخول كضيف ←</button>
    </form>
    <div class="features">
      <div class="feat"><span class="feat-dot ac"></span>محادثة ذكاء اصطناعي</div>
      <div class="feat"><span class="feat-dot ac"></span>رفع ملفات وصور</div>
      <div class="feat"><span class="feat-dot ac"></span>حفظ المحادثات محلياً</div>
      <div class="feat"><span class="feat-dot off"></span><span style="text-decoration:line-through">GitHub — متاح للمستخدم الرئيسي</span></div>
    </div>
  </div>
</div>
</body>
</html>"""

@app.get("/login", response_class=HTMLResponse)
async def login_page(err: str = ""):
    admin_err = "❌ كلمة المرور غلط" if err == "1" else ""
    return HTMLResponse(LOGIN_PAGE.format(admin_err=admin_err))

@app.post("/login/admin")
async def login_admin(password: str = Form(...)):
    if APP_PASSWORD and password != APP_PASSWORD:
        return RedirectResponse(url="/login?err=1", status_code=302)
    token = secrets.token_urlsafe(32)
    with get_db() as conn:
        conn.execute("INSERT INTO sessions (token, role) VALUES (?, ?)", (token, "admin"))
    resp  = RedirectResponse(url="/", status_code=302)
    resp.set_cookie("leo_session", token, max_age=30*24*3600, httponly=True, samesite="lax")
    resp.set_cookie("leo_role",    "admin", max_age=30*24*3600, samesite="lax")
    return resp

@app.post("/login/guest")
async def login_guest():
    token = secrets.token_urlsafe(32)
    with get_db() as conn:
        conn.execute("INSERT INTO sessions (token, role) VALUES (?, ?)", (token, "guest"))
    resp  = RedirectResponse(url="/", status_code=302)
    resp.set_cookie("leo_session", token, max_age=30*24*3600, httponly=True, samesite="lax")
    resp.set_cookie("leo_role",    "guest", max_age=30*24*3600, samesite="lax")
    return resp

@app.get("/logout")
async def logout(leo_session: str | None = Cookie(default=None)):
    with get_db() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (leo_session or "",))
    resp = RedirectResponse(url="/login", status_code=302)
    resp.delete_cookie("leo_session")
    resp.delete_cookie("leo_role")
    return resp

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, leo_session: str | None = Cookie(default=None)):
    if r := check_auth(leo_session): return r
    role = get_role(leo_session)
    return templates.TemplateResponse("index.html", {
        "request":          request,
        "models":           MODELS,
        "default_model":    MODELS[0]["id"],
        "has_github_token": bool(GITHUB_TOKEN),
        "role":             role,
        "is_admin":         role == "admin",
    })

# ── Chat Management API ──────────────────────────────────────────────────────
@app.post("/api/chats/create")
async def api_chat_create(title: str = Form(""), leo_session: str | None = Cookie(default=None)):
    if r := check_auth(leo_session): return r
    chat_id = str(uuid.uuid4())[:8]
    with get_db() as conn:
        conn.execute("INSERT INTO chats (id, title, role, user_token) VALUES (?, ?, ?, ?)",
                     (chat_id, title or None, get_role(leo_session), leo_session))
    return JSONResponse({"id": chat_id, "title": title or "محادثة جديدة"})

@app.get("/api/chats/list")
async def api_chat_list(leo_session: str | None = Cookie(default=None)):
    if r := check_auth(leo_session): return r
    chats = get_user_chats(leo_session)
    return JSONResponse({"chats": chats})

@app.post("/api/chats/rename")
async def api_chat_rename(chat_id: str = Form(...), title: str = Form(...), leo_session: str | None = Cookie(default=None)):
    if r := check_auth(leo_session): return r
    with get_db() as conn:
        conn.execute("UPDATE chats SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ? AND user_token = ?",
                     (title, chat_id, leo_session))
    return JSONResponse({"ok": True})

@app.post("/api/chats/delete")
async def api_chat_delete(chat_id: str = Form(...), leo_session: str | None = Cookie(default=None)):
    if r := check_auth(leo_session): return r
    with get_db() as conn:
        conn.execute("DELETE FROM chats WHERE id = ? AND user_token = ?", (chat_id, leo_session))
    return JSONResponse({"ok": True})

@app.get("/api/chats/{chat_id}/messages")
async def api_chat_messages(chat_id: str, leo_session: str | None = Cookie(default=None)):
    if r := check_auth(leo_session): return r
    messages = get_chat_messages(chat_id)
    return JSONResponse({"messages": messages})

# ── GitHub (admin only) ──────────────────────────────────────────────────────
@app.get("/github/repos")
async def github_repos(page:int=1, leo_session:str|None=Cookie(default=None)):
    if r := check_admin(leo_session): return r
    try:
        async with httpx.AsyncClient(timeout=15, headers=gh_headers()) as client:
            resp=await client.get("https://api.github.com/user/repos",
                params={"sort":"updated","per_page":30,"page":page,"type":"all"})
            if resp.status_code!=200:
                return JSONResponse({"error":f"GitHub: {resp.status_code}"},status_code=resp.status_code)
            return JSONResponse([{"name":r["name"],"full_name":r["full_name"],
                "description":r.get("description") or "","private":r["private"],
                "language":r.get("language") or "—","updated_at":r["updated_at"][:10],
                "stars":r["stargazers_count"],"url":r["html_url"],
                "default_branch":r.get("default_branch","main")} for r in resp.json()])
    except Exception as e: return JSONResponse({"error":str(e)},status_code=500)

@app.get("/github/tree")
async def github_tree(owner:str, repo:str, branch:str="main",
                       leo_session:str|None=Cookie(default=None)):
    if r := check_admin(leo_session): return r
    try:
        async with httpx.AsyncClient(timeout=20, headers=gh_headers()) as client:
            repo_resp=await client.get(f"https://api.github.com/repos/{owner}/{repo}")
            if repo_resp.status_code==404:
                return JSONResponse({"error":"الـ repo غير موجود"},status_code=404)
            repo_data=repo_resp.json(); branch=repo_data.get("default_branch",branch)
            tree_resp=await client.get(f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1")
            if tree_resp.status_code!=200:
                return JSONResponse({"error":"فشل جلب الشجرة"},status_code=500)
            files=[]
            for item in tree_resp.json().get("tree",[]):
                if item["type"]!="blob": continue
                path=item["path"]; parts=path.split("/")
                if any(p in SKIP_DIRS for p in parts): continue
                ext=os.path.splitext(path)[1].lower()
                files.append({"path":path,"size":item.get("size",0),"is_code":ext in CODE_EXT,"ext":ext})
            return JSONResponse({"repo":f"{owner}/{repo}","branch":branch,"files":files,"total":len(files),
                "description":repo_data.get("description") or "","language":repo_data.get("language") or "—",
                "stars":repo_data.get("stargazers_count",0)})
    except Exception as e: return JSONResponse({"error":str(e)},status_code=500)

@app.get("/github/file")
async def github_file(owner:str, repo:str, path:str, branch:str="main",
                       leo_session:str|None=Cookie(default=None)):
    if r := check_admin(leo_session): return r
    try:
        async with httpx.AsyncClient(timeout=15, headers=gh_headers()) as client:
            resp=await client.get(f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}")
            if resp.status_code==404: return JSONResponse({"error":"الملف غير موجود"},status_code=404)
            content=resp.text
            return JSONResponse({"path":path,"content":content[:20_000],
                "lines":len(content.splitlines()),"size":len(content),"truncated":len(content)>20_000})
    except Exception as e: return JSONResponse({"error":str(e)},status_code=500)

@app.post("/github/load-to-memory")
async def github_load(owner:str=Form(...), repo:str=Form(...), branch:str=Form("main"),
                       paths:str=Form(""), leo_session:str|None=Cookie(default=None)):
    if r := check_admin(leo_session): return r
    path_list=[p.strip() for p in paths.split(",") if p.strip()][:30]
    added=[]; errors=[]
    async with httpx.AsyncClient(timeout=20, headers=gh_headers()) as client:
        for path in path_list:
            try:
                resp=await client.get(f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}")
                if resp.status_code==200:
                    with get_db() as conn:
                        conn.execute("""INSERT INTO project_files (user_token, filename, content) 
                                       VALUES (?, ?, ?) 
                                       ON CONFLICT(user_token, filename) DO UPDATE SET content=excluded.content""",
                                    (leo_session, f"{repo}/{path}", resp.text[:10_000]))
                    added.append(path)
                else: errors.append(f"{path}: HTTP {resp.status_code}")
            except Exception as e: errors.append(f"{path}: {e}")
    return JSONResponse({"ok":True,"added":added,"errors":errors})

@app.post("/github/review")
async def github_review(owner:str=Form(...), repo:str=Form(...), branch:str=Form("main"),
                         paths:str=Form(""), model_id:str=Form(None), chat_id:str=Form("default"),
                         leo_session:str|None=Cookie(default=None)):
    if r := check_admin(leo_session): return r
    if not OPENROUTER_API_KEY:
        async def _e():
            yield f"data: {json.dumps({'error':'API key not set'})}\n\n"
        return StreamingResponse(_e(), media_type="text/event-stream")

    path_list=[p.strip() for p in paths.split(",") if p.strip()][:20]
    file_contents:dict[str,str]={}
    async with httpx.AsyncClient(timeout=20, headers=gh_headers()) as client:
        for path in path_list:
            try:
                resp=await client.get(f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}")
                if resp.status_code==200: file_contents[path]=resp.text[:8_000]
            except: pass

    if not file_contents:
        async def _e():
            yield f"data: {json.dumps({'error':'لم يتم جلب أي ملف'})}\n\n"
        return StreamingResponse(_e(), media_type="text/event-stream")

    files_block="\n\n".join(f"### {p}\n```\n{c}\n```" for p,c in file_contents.items())
    review_prompt=f"""قم بمراجعة شاملة للكود التالي من {owner}/{repo}:

{files_block}

المطلوب:
1. **تقييم عام** /10 مع السبب
2. **🐛 الأخطاء** (مع رقم السطر)
3. **🔐 المشاكل الأمنية** (CRITICAL/HIGH/MEDIUM/LOW)
4. **⚡ مشاكل الأداء**
5. **🏗️ مشاكل المعمارية**
6. **📝 جودة الكود**
7. **✅ الكود المُصحح** لكل مشكلة
8. **💡 اقتراحات التحسين**"""

    valid_ids=[m["id"] for m in MODELS]
    chosen=model_id if model_id in valid_ids else MODELS[0]["id"]
    ordered=[chosen]+[mid for mid in CODING_IDS if mid!=chosen]

    messages_history = get_chat_messages(chat_id)[-10:]

    stop_ev=asyncio.Event(); stop_flags[chat_id]=stop_ev
    messages=[{"role":"system","content":SYSTEM_PROMPT_ADMIN}]+messages_history+[{"role":"user","content":review_prompt}]
    hdrs={"Authorization":f"Bearer {OPENROUTER_API_KEY}","Content-Type":"application/json","HTTP-Referer":"https://leo-ai.app","X-Title":"Leo-AI"}

    async def streamer():
        full_text=""; used_model=ordered[0]; last_err=""
        for mid in ordered:
            if stop_ev.is_set(): break
            try:
                to=httpx.Timeout(connect=8.0,read=90.0,write=10.0,pool=5.0)
                async with httpx.AsyncClient(timeout=to) as client:
                    async with client.stream("POST","https://openrouter.ai/api/v1/chat/completions",
                        headers=hdrs,json={"model":mid,"messages":messages,"temperature":0.2,"max_tokens":4000,"stream":True}) as resp:
                        if resp.status_code!=200:
                            body=await resp.aread()
                            try: last_err=json.loads(body).get("error",{}).get("message","")
                            except: last_err=body.decode(errors="replace")[:200]
                            continue
                        used_model=mid; buf=""; got_first=False
                        async for chunk in resp.aiter_text():
                            if stop_ev.is_set(): break
                            buf+=chunk; lines=buf.split("\n"); buf=lines.pop()
                            for line in lines:
                                line=line.strip()
                                if not line.startswith("data:"): continue
                                ds=line[5:].strip()
                                if ds=="[DONE]": break
                                try:
                                    delta=json.loads(ds)["choices"][0]["delta"].get("content","")
                                    if delta:
                                        if not got_first: got_first=True; yield f"data: {json.dumps({'first':True})}\n\n"
                                        full_text+=delta; yield f"data: {json.dumps({'token':delta})}\n\n"
                                except: continue
                break
            except: continue
        if not full_text: full_text=f"⚠️ فشل: {last_err}"; yield f"data: {json.dumps({'token':full_text})}\n\n"

        with get_db() as conn:
            conn.execute("INSERT INTO messages (chat_id, role, content, model) VALUES (?, ?, ?, ?)",
                        (chat_id, "user", review_prompt, chosen))
            conn.execute("INSERT INTO messages (chat_id, role, content, model) VALUES (?, ?, ?, ?)",
                        (chat_id, "assistant", full_text, used_model))

        used_name=next((m["name"] for m in MODELS if m["id"]==used_model),"")
        yield f"data: {json.dumps({'done':True,'html':markdown_to_html(full_text),'used_model':used_name,'files':{},'raw':full_text})}\n\n"

    return StreamingResponse(streamer(),media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

# ── Project memory (admin only) ──────────────────────────────────────────────
@app.post("/project/add")
async def project_add(files:List[UploadFile]=File(default=[]), leo_session:str|None=Cookie(default=None)):
    if r := check_admin(leo_session): return r
    added=[]
    for f in files:
        if not f or not f.filename: continue
        try:
            raw=await f.read(); content=read_file(f.filename,raw,f.content_type or "")
            with get_db() as conn:
                conn.execute("""INSERT INTO project_files (user_token, filename, content) 
                               VALUES (?, ?, ?) 
                               ON CONFLICT DO UPDATE SET content=excluded.content""",
                            (leo_session, f.filename, content))
            added.append(f.filename)
        except Exception as e: logger.error(f"Project {f.filename}: {e}")
    return JSONResponse({"ok":True,"added":added})

@app.get("/project/list")
async def project_list(leo_session:str|None=Cookie(default=None)):
    if r := check_admin(leo_session): return r
    with get_db() as conn:
        rows = conn.execute("SELECT filename, LENGTH(content) as size FROM project_files WHERE user_token = ?", (leo_session,)).fetchall()
    return JSONResponse({"files":[{"name":r["filename"],"size":r["size"]} for r in rows]})

@app.post("/project/remove")
async def project_remove(filename:str=Form(...), leo_session:str|None=Cookie(default=None)):
    if r := check_admin(leo_session): return r
    with get_db() as conn:
        conn.execute("DELETE FROM project_files WHERE user_token = ? AND filename = ?", (leo_session, filename))
    return JSONResponse({"ok":True})

@app.post("/project/clear")
async def project_clear(leo_session:str|None=Cookie(default=None)):
    if r := check_admin(leo_session): return r
    with get_db() as conn:
        conn.execute("DELETE FROM project_files WHERE user_token = ?", (leo_session,))
    return JSONResponse({"ok":True})

# ── Chat stream (both roles) ─────────────────────────────────────────────────
@app.post("/chat-stream")
async def chat_stream(request:Request, prompt:str=Form(...), model_id:str=Form(None),
                       files:List[UploadFile]=File(default=[]), chat_id:str=Form("default"),
                       leo_session:str|None=Cookie(default=None)):
    if r := check_auth(leo_session): return r
    if not OPENROUTER_API_KEY:
        async def _e():
            yield f"data: {json.dumps({'error':'OPENROUTER_API_KEY not set'})}\n\n"
        return StreamingResponse(_e(),media_type="text/event-stream")

    role=get_role(leo_session)

    valid_ids=[m["id"] for m in MODELS]
    chosen=model_id if model_id in valid_ids else MODELS[0]["id"]
    user_text=prompt; img_parts=[]; has_image=False; switched=False

    real_files=[f for f in (files or []) if f and f.filename][:25]
    for f in real_files:
        try:
            raw=await f.read(); mt=f.content_type or ""
            if mt.startswith("image/"):
                has_image=True; b64=base64.b64encode(raw).decode()
                img_parts.append({"type":"image_url","image_url":{"url":f"data:{mt};base64,{b64}"}})
            else: user_text+=f"\n\n{read_file(f.filename,raw,mt)}"
        except Exception as e: logger.error(f"File {f.filename}: {e}")

    if has_image:
        obj=next((m for m in MODELS if m["id"]==chosen),None)
        if not (obj and obj["vision"]): chosen=VISION_IDS[0] if VISION_IDS else chosen; switched=True

    ordered=VISION_IDS if has_image else ([chosen]+[mid for mid in CODING_IDS if mid!=chosen])

    messages_history = get_chat_messages(chat_id)[-10:]

    stop_ev=asyncio.Event(); stop_flags[chat_id]=stop_ev

    sys_prompt=SYSTEM_PROMPT_ADMIN if role=="admin" else SYSTEM_PROMPT_GUEST
    if role=="admin":
        project_ctx=build_project_context(leo_session)
        if project_ctx: sys_prompt+="\n\n"+project_ctx

    content_parts=[{"type":"text","text":user_text}]+img_parts
    msg_content=content_parts if (img_parts or len(content_parts)>1) else user_text
    messages=[{"role":"system","content":sys_prompt}]+messages_history+[{"role":"user","content":msg_content}]
    hdrs={"Authorization":f"Bearer {OPENROUTER_API_KEY}","Content-Type":"application/json",
          "HTTP-Referer":"https://leo-ai.app","X-Title":"Leo-AI"}

    is_first = len(messages_history) == 0
    auto_title = generate_title(prompt) if is_first else None

    async def streamer():
        full_text=""; used_model=ordered[0] if ordered else chosen; last_err=""
        for mid in ordered:
            if stop_ev.is_set(): break
            try:
                to=httpx.Timeout(connect=8.0,read=90.0,write=10.0,pool=5.0)
                async with httpx.AsyncClient(timeout=to) as client:
                    async with client.stream("POST","https://openrouter.ai/api/v1/chat/completions",
                        headers=hdrs,json={"model":mid,"messages":messages,"temperature":0.3,"max_tokens":4000,"stream":True}) as resp:
                        if resp.status_code!=200:
                            body=await resp.aread()
                            try: last_err=json.loads(body).get("error",{}).get("message","")
                            except: last_err=body.decode(errors="replace")[:200]
                            continue
                        used_model=mid; buf=""; got_first=False
                        async for chunk in resp.aiter_text():
                            if stop_ev.is_set(): break
                            buf+=chunk; lines=buf.split("\n"); buf=lines.pop()
                            for line in lines:
                                line=line.strip()
                                if not line.startswith("data:"): continue
                                ds=line[5:].strip()
                                if ds=="[DONE]": break
                                try:
                                    delta=json.loads(ds)["choices"][0]["delta"].get("content","")
                                    if delta:
                                        if not got_first: got_first=True; yield f"data: {json.dumps({'first':True})}\n\n"
                                        full_text+=delta; yield f"data: {json.dumps({'token':delta})}\n\n"
                                except: continue
                break
            except httpx.ReadTimeout: last_err="timeout"; continue
            except asyncio.CancelledError: break
            except Exception as exc: last_err=str(exc); continue

        if stop_ev.is_set() and full_text: full_text+="\n\n*(⏹ توقف)*"
        if not full_text: full_text=f"⚠️ فشلت كل النماذج.\n\nالخطأ: `{last_err}`"; yield f"data: {json.dumps({'token':full_text})}\n\n"

        with get_db() as conn:
            conn.execute("INSERT INTO messages (chat_id, role, content, model) VALUES (?, ?, ?, ?)",
                        (chat_id, "user", prompt, chosen))
            conn.execute("INSERT INTO messages (chat_id, role, content, model) VALUES (?, ?, ?, ?)",
                        (chat_id, "assistant", full_text, used_model))

            if auto_title:
                conn.execute("UPDATE chats SET title = ? WHERE id = ?", (auto_title, chat_id))

            conn.execute("UPDATE chats SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (chat_id,))

        files_found=extract_files(full_text)
        with get_db() as conn:
            for fname, content in files_found.items():
                conn.execute("""INSERT INTO generated_files (chat_id, filename, content) VALUES (?, ?, ?)
                               ON CONFLICT(chat_id, filename) DO UPDATE SET content=excluded.content""",
                            (chat_id, fname, content))

        files_b64={n:base64.b64encode(c.encode()).decode() for n,c in files_found.items()}
        used_name=next((m["name"] for m in MODELS if m["id"]==used_model),"")
        yield f"data: {json.dumps({'done':True,'html':markdown_to_html(full_text),'files':files_b64,'used_model':used_name,'switched':switched,'raw':full_text,'role':role,'auto_title':auto_title})}\n\n"

    return StreamingResponse(streamer(),media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

# ── Common routes ──────────────────────────────────────────────────────────────
@app.post("/stop")
async def stop(chat_id:str=Form("default"),leo_session:str|None=Cookie(default=None)):
    if r := check_auth(leo_session): return r
    ev=stop_flags.get(chat_id)
    if ev: ev.set()
    return JSONResponse({"ok":True})

@app.post("/download")
async def download(filename:str=Form(...),code_b64:str=Form(...),
                    leo_session:str|None=Cookie(default=None)):
    if r := check_auth(leo_session): return r
    try: code=base64.b64decode(code_b64).decode("utf-8")
    except: return JSONResponse({"error":"invalid"},status_code=400)
    ext=filename.rsplit(".",1)[-1].lower()
    mime={"py":"text/x-python","js":"application/javascript","ts":"text/typescript",
          "html":"text/html","css":"text/css","json":"application/json","txt":"text/plain",
          "md":"text/markdown","sh":"text/x-sh","go":"text/x-go","rs":"text/x-rust"}
    return Response(content=code,media_type=mime.get(ext,"text/plain"),
        headers={"Content-Disposition":f'attachment; filename="{filename}"',
                 "Content-Type":f'{mime.get(ext,"text/plain")}; charset=utf-8'})

@app.get("/download/zip/{chat_id}")
async def download_zip(chat_id:str="default",leo_session:str|None=Cookie(default=None)):
    if r := check_auth(leo_session): return r
    with get_db() as conn:
        rows = conn.execute("SELECT filename, content FROM generated_files WHERE chat_id = ?", (chat_id,)).fetchall()
    if not rows: return JSONResponse({"error":"لا توجد ملفات"},status_code=404)
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as zf:
        for r in rows: zf.writestr(r["filename"],r["content"])
    buf.seek(0)
    return Response(content=buf.read(),media_type="application/zip",
        headers={"Content-Disposition":'attachment; filename="leo-ai.zip"'})

@app.post("/clear")
async def clear(chat_id:str=Form("default"),leo_session:str|None=Cookie(default=None)):
    if r := check_auth(leo_session): return r
    with get_db() as conn:
        conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        conn.execute("DELETE FROM generated_files WHERE chat_id = ?", (chat_id,))
    ev=stop_flags.get(chat_id)
    if ev: ev.set()
    return JSONResponse({"ok":True})

@app.post("/share")
async def share(html:str=Form(...),title:str=Form("محادثة"),
                 leo_session:str|None=Cookie(default=None)):
    if r := check_auth(leo_session): return r
    sid=hashlib.md5(f"{title}{time.time()}".encode()).hexdigest()[:8]
    shared_chats[sid]={"html":html,"title":title,"created":time.time()}
    return JSONResponse({"id":sid,"url":f"/share/{sid}"})

@app.get("/share/{sid}",response_class=HTMLResponse)
async def view_share(sid:str):
    chat=shared_chats.get(sid)
    if not chat: return HTMLResponse("<h2>غير موجود</h2>",status_code=404)
    return HTMLResponse(f"""<!DOCTYPE html><html lang="ar" dir="rtl">
<head><meta charset="UTF-8"><title>{chat['title']} — Leo-AI</title>
<style>body{{font-family:'Segoe UI',sans-serif;background:#212121;color:#ececec;padding:20px;max-width:800px;margin:0 auto}}
.cw2{{background:#1a1a1a;border-radius:9px;margin:8px 0;border:1px solid #404040;overflow:hidden}}
.ch{{display:flex;padding:6px 12px;background:#252525;border-bottom:1px solid #404040}}
.ch span{{color:#8e8ea0;font-size:11px;font-family:monospace}}pre{{margin:0;padding:12px;overflow-x:auto;font-size:12px;color:#cdd6f4}}
h1,h2,h3{{color:#19c37d}}strong{{color:#fff}}
.row{{display:flex;gap:10px;margin:8px 0;padding:4px}}.row.u{{justify-content:flex-end}}
.bbl{{padding:9px 13px;border-radius:12px;font-size:14px;line-height:1.7;max-width:82%}}
.bbl.u{{background:#2f2f2f;border:1px solid #404040}}.bbl.a{{background:transparent;width:100%;max-width:100%}}
.av{{width:26px;height:26px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:12px;flex-shrink:0;margin-top:3px}}
.av.aa{{background:linear-gradient(135deg,#19c37d,#ab68ff);order:-1}}.av.ua{{background:linear-gradient(135deg,#ab68ff,#6b3fa0)}}
</style></head><body>
<div style="background:#171717;padding:12px 16px;border-radius:10px;margin-bottom:20px;display:flex;justify-content:space-between">
<span style="color:#19c37d;font-weight:700">🚀 Leo-AI — {chat['title']}</span>
<span style="color:#8e8ea0;font-size:12px">محادثة مشتركة</span></div>
{chat['html']}</body></html>""")

@app.post("/generate-image")
async def generate_image(prompt:str=Form(...),size:str=Form("1024x1024"),model:str=Form("flux"),
                          leo_session:str|None=Cookie(default=None)):
    if r := check_auth(leo_session): return r
    try:
        w,h=map(int,size.split("x")) if "x" in size else (1024,1024)
        return JSONResponse({"url":img_url(prompt,w,h,model),"prompt":prompt})
    except Exception as e: return JSONResponse({"error":str(e)},status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

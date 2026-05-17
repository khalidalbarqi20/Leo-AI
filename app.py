"""
Leo-AI v6 — with password authentication
"""

import os, base64, logging, re, zipfile, tarfile, io, json
import urllib.parse, asyncio, hashlib, time, secrets
import httpx

from fastapi import FastAPI, Request, Form, UploadFile, File, Cookie
from fastapi.responses import HTMLResponse, Response, JSONResponse, StreamingResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from typing import List, Optional

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("leo-ai")

app = FastAPI(title="Leo-AI", version="6.1.0")
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY")
GITHUB_TOKEN:       str | None = os.getenv("GITHUB_TOKEN")
APP_PASSWORD:       str | None = os.getenv("APP_PASSWORD", "")

# ── Auth: valid session tokens ─────────────────────────────────────────────────
# In-memory set of valid session tokens (cleared on restart)
valid_sessions: set[str] = set()

def is_authenticated(session_token: str | None) -> bool:
    """Check if the session token is valid."""
    if not APP_PASSWORD:
        return True   # no password set → open access
    return session_token in valid_sessions

def require_auth(session_token: str | None):
    """Return RedirectResponse if not authenticated, else None."""
    if not is_authenticated(session_token):
        return RedirectResponse(url="/login", status_code=302)
    return None

# ── Session state ──────────────────────────────────────────────────────────────
chat_sessions:   dict[str, list]          = {}
project_memory:  dict[str, dict]          = {}
stop_flags:      dict[str, asyncio.Event] = {}
generated_files: dict[str, dict]          = {}
shared_chats:    dict[str, dict]          = {}

def gh_headers() -> dict:
    h = {"Accept": "application/vnd.github.v3+json", "User-Agent": "Leo-AI/6.1"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"token {GITHUB_TOKEN}"
    return h

SYSTEM_PROMPT = """You are a senior software engineer, system architect, debugging specialist, security reviewer, and production-grade AI coding agent.

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
- Report: 🐛 Bugs, 🔐 Security vulnerabilities, ⚡ Performance issues, 🏗️ Architecture problems, 📝 Code quality issues
- Provide specific line references when possible
- Show the fixed code for every issue found
- Rate overall code quality: /10
- Prioritize: CRITICAL > HIGH > MEDIUM > LOW

FORMAT:
- Detect user language and respond in SAME language
- Write code in Markdown code blocks with language tags
- For multi-file projects: ### filename.ext before each block
"""

MODELS: list[dict] = [
    {"id":"qwen/qwen3-coder-480b-a35b-instruct:free","name":"Qwen3 Coder 480B","desc":"الأقوى للكود • 262K","badge":"🥇","context":"262K","speed":"متوسط","tag":"أفضل للكود","vision":False,"strength":"code"},
    {"id":"deepseek/deepseek-r1:free","name":"DeepSeek R1","desc":"تفكير عميق • debugging","badge":"🧠","context":"128K","speed":"بطيء","tag":"تفكير عميق","vision":False,"strength":"reasoning"},
    {"id":"moonshotai/kimi-k2:free","name":"Kimi K2","desc":"بناء المشاريع الكاملة","badge":"🚀","context":"128K","speed":"متوسط","tag":"مشاريع","vision":False,"strength":"projects"},
    {"id":"qwen/qwen2.5-vl-72b-instruct:free","name":"Qwen2.5 VL 72B","desc":"يقرأ الصور","badge":"👁️","context":"128K","speed":"متوسط","tag":"يرى الصور","vision":True,"strength":"vision"},
    {"id":"qwen/qwen2.5-vl-32b-instruct:free","name":"Qwen2.5 VL 32B","desc":"يقرأ الصور • سريع","badge":"🔍","context":"128K","speed":"سريع","tag":"يرى الصور","vision":True,"strength":"vision"},
    {"id":"google/gemma-4-31b-it:free","name":"Gemma 4 31B","desc":"يقرأ الصور • 256K","badge":"🌟","context":"256K","speed":"متوسط","tag":"يرى الصور","vision":True,"strength":"vision"},
    {"id":"openai/gpt-oss-20b:free","name":"GPT-OSS 20B","desc":"سريع ودقيق","badge":"⚡","context":"128K","speed":"سريع","tag":"سريع","vision":False,"strength":"general"},
    {"id":"meta-llama/llama-4-scout:free","name":"Llama 4 Scout","desc":"الأسرع","badge":"⚡","context":"128K","speed":"سريع جداً","tag":"الأسرع","vision":False,"strength":"general"},
]
VISION_IDS = [m["id"] for m in MODELS if m["vision"]]
CODING_IDS = [m["id"] for m in MODELS if not m["vision"]]

CODE_EXT = {
    ".py",".js",".ts",".tsx",".jsx",".html",".css",".json",
    ".md",".txt",".yml",".yaml",".sh",".bash",".go",".rs",
    ".java",".cpp",".c",".h",".cs",".rb",".php",".vue",
    ".sql",".toml",".env.example",".gitignore",".dockerfile",
    ".tf",".kt",".swift",".r",".scala",
}
SKIP_DIRS = {
    "node_modules",".git","dist","build","__pycache__",
    ".venv","venv","env",".next",".nuxt","coverage",
    ".pytest_cache",".mypy_cache","target","vendor",
}
MAX_FC = 15_000

# ── Markdown → HTML ────────────────────────────────────────────────────────────
def markdown_to_html(text: str) -> str:
    code_blocks: list[tuple[str,str]] = []
    def extract(m):
        lang = m.group(1) or "text"
        code = m.group(2)
        idx  = len(code_blocks)
        code_blocks.append((lang, code))
        return f"___CODE_{idx}___"
    text = re.sub(r"```(\w+)?\n(.*?)```", extract, text, flags=re.DOTALL)
    text = text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"^### (.*?)$", r"<h3>\1</h3>", text, flags=re.MULTILINE)
    text = re.sub(r"^## (.*?)$",  r"<h2>\1</h2>", text, flags=re.MULTILINE)
    text = re.sub(r"^# (.*?)$",   r"<h1>\1</h1>", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.*?)\*",     r"<em>\1</em>", text)
    text = re.sub(r"^> (.*?)$", r"<blockquote>\1</blockquote>", text, flags=re.MULTILINE)
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
        esc_code = code.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        b64 = base64.b64encode(code.encode()).decode()
        html = (f'<div class="cw2"><div class="ch"><span>{lang or "code"}</span>'
                f'<div class="ch-btns"><button onclick="cpy(this,\'{b64}\')">📋 نسخ</button>'
                f'<button onclick="explainCode(this,\'{b64}\')">💡 شرح</button></div></div>'
                f'<pre><code class="lang-{lang}">{esc_code}</code></pre></div>')
        text = text.replace(f"___CODE_{idx}___", html)
    return text

def extract_files(text: str) -> dict[str,str]:
    pat = re.compile(
        r"###\s*(\S+\.(?:py|html|css|js|json|txt|md|jsx|ts|tsx|vue|php|"
        r"java|cpp|c|go|rs|rb|sh|sql|xml|yaml|yml))\s*\n*```(?:\w+)?\n(.*?)```",
        re.DOTALL | re.IGNORECASE,
    )
    files = {f.strip(): c.strip() for f,c in pat.findall(text)}
    if not files:
        m = re.search(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
        if m:
            files[_guess(m.group(1))] = m.group(1).strip()
    return files

def _guess(c):
    cl = c.lower().strip()
    if cl.startswith(("<!doctype","<html")):          return "index.html"
    if "fastapi" in cl or cl.startswith("import "): return "app.py"
    if cl.startswith(("const ","let ","function ")):  return "script.js"
    if "body {" in cl:                               return "style.css"
    if cl.startswith(("{","[")):                     return "data.json"
    return "code.txt"

def read_file(fname, raw, ct):
    nl = fname.lower()
    if nl.endswith(".pdf"): return _read_pdf(raw, fname)
    if nl.endswith(".zip") or "zip" in ct:
        parts = [f"[ZIP: {fname}]"]
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                for i in zf.infolist():
                    if i.is_dir() or i.file_size > 500_000: continue
                    try:
                        data = zf.read(i.filename)
                        parts.append(f"\n--- {i.filename} ---\n{data.decode('utf-8',errors='replace')[:3000]}")
                    except: parts.append(f"\n--- {i.filename} --- (binary)")
        except Exception as e: parts.append(f"(error: {e})")
        return "\n".join(parts)[:MAX_FC]
    try:    return f"[{fname}]\n{raw.decode('utf-8',errors='replace')}"[:MAX_FC]
    except: return f"(cannot read: {fname})"

def _read_pdf(raw, fname):
    if not HAS_PYPDF: return f"[PDF: {fname}] — pypdf not installed"
    try:
        reader = PdfReader(io.BytesIO(raw))
        pages  = []
        for i,page in enumerate(reader.pages[:50]):
            t = page.extract_text() or ""
            if t.strip(): pages.append(f"[صفحة {i+1}]\n{t.strip()}")
        return f"[PDF: {fname}]\n\n" + "\n\n".join(pages)[:MAX_FC]
    except Exception as e:
        return f"[PDF: {fname}] error: {e}"

def build_project_context(sid):
    mem = project_memory.get(sid, {})
    if not mem: return ""
    parts = ["=== PROJECT CONTEXT ==="]
    for fname,content in mem.items():
        parts.append(f"\n### {fname}\n{content[:3000]}")
    parts.append("=== END PROJECT CONTEXT ===\n")
    return "\n".join(parts)

def img_url(p,w=1024,h=1024,model="flux"):
    return (f"https://image.pollinations.ai/prompt/{urllib.parse.quote(p)}"
            f"?width={w}&height={h}&model={model}&nologo=true&seed={abs(hash(p))%99999}")

# ══════════════════════════════════════════════════════════════════════════════
# AUTH ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Show login page."""
    return HTMLResponse("""<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<meta name="theme-color" content="#171717">
<title>Leo-AI — تسجيل الدخول</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#212121;color:#ececec;
     display:flex;align-items:center;justify-content:center;min-height:100vh;padding:20px}
.card{background:#2f2f2f;border:1px solid #404040;border-radius:16px;
      padding:32px 24px;width:100%;max-width:340px;text-align:center}
.logo{width:64px;height:64px;background:linear-gradient(135deg,#19c37d,#ab68ff);
      border-radius:50%;display:flex;align-items:center;justify-content:center;
      font-size:30px;margin:0 auto 16px}
h1{font-size:20px;font-weight:700;margin-bottom:6px}
p{color:#8e8ea0;font-size:13px;margin-bottom:24px}
input{width:100%;background:#1a1a1a;color:#ececec;border:1px solid #404040;
      border-radius:10px;padding:13px 14px;font-size:15px;outline:none;
      text-align:center;letter-spacing:2px;margin-bottom:14px;font-family:inherit}
input:focus{border-color:#19c37d}
button{width:100%;background:#19c37d;color:#111;border:none;border-radius:10px;
       padding:13px;font-size:15px;font-weight:700;cursor:pointer;transition:opacity .15s}
button:active{opacity:.85}
.err{color:#ef4444;font-size:12px;margin-top:10px;min-height:18px}
</style>
</head>
<body>
<div class="card">
  <div class="logo">🚀</div>
  <h1>Leo-AI</h1>
  <p>أدخل كلمة المرور للمتابعة</p>
  <form method="post" action="/login">
    <input type="password" name="password" placeholder="••••••••••"
           autocomplete="current-password" autofocus>
    <button type="submit">دخول ←</button>
  </form>
  <div class="err" id="err"></div>
</div>
<script>
// Show error if redirected with ?err=1
if(new URLSearchParams(location.search).get('err')==='1'){
  document.getElementById('err').textContent='❌ كلمة المرور غلط، حاول مرة ثانية';
}
</script>
</body>
</html>""")

@app.post("/login")
async def login(password: str = Form(...)):
    """Verify password and set session cookie."""
    if not APP_PASSWORD or password == APP_PASSWORD:
        token = secrets.token_urlsafe(32)
        valid_sessions.add(token)
        response = RedirectResponse(url="/", status_code=302)
        # Cookie valid for 30 days
        response.set_cookie(
            key="leo_session",
            value=token,
            max_age=30 * 24 * 3600,
            httponly=True,
            samesite="lax",
        )
        return response
    return RedirectResponse(url="/login?err=1", status_code=302)

@app.get("/logout")
async def logout(leo_session: str | None = Cookie(default=None)):
    valid_sessions.discard(leo_session or "")
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie("leo_session")
    return response

# ══════════════════════════════════════════════════════════════════════════════
# MAIN ROUTES (protected)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, leo_session: str | None = Cookie(default=None)):
    redirect = require_auth(leo_session)
    if redirect: return redirect
    return templates.TemplateResponse("index.html", {
        "request": request,
        "models":  MODELS,
        "default_model": MODELS[0]["id"],
        "has_github_token": bool(GITHUB_TOKEN),
    })

# ── GitHub API ─────────────────────────────────────────────────────────────────
@app.get("/github/repos")
async def github_repos(page:int=1, leo_session:str|None=Cookie(default=None)):
    if r := require_auth(leo_session): return r
    if not GITHUB_TOKEN:
        return JSONResponse({"error":"GITHUB_TOKEN غير موجود"}, status_code=401)
    try:
        async with httpx.AsyncClient(timeout=15, headers=gh_headers()) as client:
            resp = await client.get("https://api.github.com/user/repos",
                params={"sort":"updated","per_page":30,"page":page,"type":"all"})
            if resp.status_code != 200:
                return JSONResponse({"error":f"GitHub: {resp.status_code}"}, status_code=resp.status_code)
            return JSONResponse([{
                "name":r["name"],"full_name":r["full_name"],
                "description":r.get("description") or "","private":r["private"],
                "language":r.get("language") or "—","updated_at":r["updated_at"][:10],
                "stars":r["stargazers_count"],"url":r["html_url"],
                "default_branch":r.get("default_branch","main"),
            } for r in resp.json()])
    except Exception as e:
        return JSONResponse({"error":str(e)}, status_code=500)

@app.get("/github/tree")
async def github_tree(owner:str, repo:str, branch:str="main", leo_session:str|None=Cookie(default=None)):
    if r := require_auth(leo_session): return r
    try:
        async with httpx.AsyncClient(timeout=20, headers=gh_headers()) as client:
            repo_resp = await client.get(f"https://api.github.com/repos/{owner}/{repo}")
            if repo_resp.status_code == 404:
                return JSONResponse({"error":"الـ repo غير موجود"}, status_code=404)
            repo_data = repo_resp.json()
            branch    = repo_data.get("default_branch", branch)
            tree_resp = await client.get(
                f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1")
            if tree_resp.status_code != 200:
                return JSONResponse({"error":f"فشل جلب الشجرة"}, status_code=500)
            files = []
            for item in tree_resp.json().get("tree",[]):
                if item["type"] != "blob": continue
                path  = item["path"]
                parts = path.split("/")
                if any(p in SKIP_DIRS for p in parts): continue
                ext = os.path.splitext(path)[1].lower()
                files.append({"path":path,"size":item.get("size",0),
                               "is_code":ext in CODE_EXT,"ext":ext})
            return JSONResponse({"repo":f"{owner}/{repo}","branch":branch,"files":files,
                                  "total":len(files),"description":repo_data.get("description") or "",
                                  "language":repo_data.get("language") or "—",
                                  "stars":repo_data.get("stargazers_count",0)})
    except Exception as e:
        return JSONResponse({"error":str(e)}, status_code=500)

@app.get("/github/file")
async def github_file(owner:str, repo:str, path:str, branch:str="main",
                       leo_session:str|None=Cookie(default=None)):
    if r := require_auth(leo_session): return r
    try:
        async with httpx.AsyncClient(timeout=15, headers=gh_headers()) as client:
            resp = await client.get(
                f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}")
            if resp.status_code == 404:
                return JSONResponse({"error":"الملف غير موجود"}, status_code=404)
            content = resp.text
            return JSONResponse({"path":path,"content":content[:20_000],
                                  "lines":len(content.splitlines()),
                                  "size":len(content),"truncated":len(content)>20_000})
    except Exception as e:
        return JSONResponse({"error":str(e)}, status_code=500)

@app.post("/github/load-to-memory")
async def github_load(owner:str=Form(...), repo:str=Form(...), branch:str=Form("main"),
                       paths:str=Form(""), sid:str=Form("default"),
                       leo_session:str|None=Cookie(default=None)):
    if r := require_auth(leo_session): return r
    path_list = [p.strip() for p in paths.split(",") if p.strip()][:30]
    project_memory.setdefault(sid, {})
    added=[]; errors=[]
    async with httpx.AsyncClient(timeout=20, headers=gh_headers()) as client:
        for path in path_list:
            try:
                resp = await client.get(
                    f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}")
                if resp.status_code == 200:
                    project_memory[sid][f"{repo}/{path}"] = \
                        f"[GitHub: {owner}/{repo}/{path}]\n{resp.text[:10_000]}"
                    added.append(path)
                else: errors.append(f"{path}: HTTP {resp.status_code}")
            except Exception as e: errors.append(f"{path}: {e}")
    return JSONResponse({"ok":True,"added":added,"errors":errors,
                          "total":len(project_memory[sid]),
                          "files":list(project_memory[sid].keys())})

@app.post("/github/review")
async def github_review(owner:str=Form(...), repo:str=Form(...), branch:str=Form("main"),
                         paths:str=Form(""), model_id:str=Form(None), sid:str=Form("default"),
                         leo_session:str|None=Cookie(default=None)):
    if r := require_auth(leo_session): return r
    if not OPENROUTER_API_KEY:
        async def _e():
            yield f"data: {json.dumps({'error':'OPENROUTER_API_KEY not set'})}\n\n"
        return StreamingResponse(_e(), media_type="text/event-stream")

    path_list = [p.strip() for p in paths.split(",") if p.strip()][:20]
    file_contents: dict[str,str] = {}
    async with httpx.AsyncClient(timeout=20, headers=gh_headers()) as client:
        for path in path_list:
            try:
                resp = await client.get(
                    f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}")
                if resp.status_code == 200:
                    file_contents[path] = resp.text[:8_000]
            except: pass

    if not file_contents:
        async def _e():
            yield f"data: {json.dumps({'error':'لم يتم جلب أي ملف'})}\n\n"
        return StreamingResponse(_e(), media_type="text/event-stream")

    files_block = "\n\n".join(
        f"### {path}\n```\n{content}\n```" for path,content in file_contents.items())
    review_prompt = f"""قم بمراجعة شاملة للكود التالي من {owner}/{repo}:

{files_block}

المطلوب:
1. **تقييم عام** /10 مع السبب
2. **🐛 الأخطاء والمشاكل** (مع رقم السطر)
3. **🔐 المشاكل الأمنية** (CRITICAL/HIGH/MEDIUM/LOW)
4. **⚡ مشاكل الأداء**
5. **🏗️ مشاكل المعمارية**
6. **📝 جودة الكود**
7. **✅ الكود المُصحح** لكل مشكلة
8. **💡 اقتراحات التحسين**
"""
    valid_ids = [m["id"] for m in MODELS]
    chosen    = model_id if model_id in valid_ids else MODELS[0]["id"]
    ordered   = [chosen] + [mid for mid in CODING_IDS if mid != chosen]
    chat_sessions.setdefault(sid, [])
    stop_ev = asyncio.Event(); stop_flags[sid] = stop_ev
    messages = [{"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":review_prompt}]
    hdrs = {"Authorization":f"Bearer {OPENROUTER_API_KEY}","Content-Type":"application/json",
            "HTTP-Referer":"https://leo-ai.app","X-Title":"Leo-AI"}

    async def streamer():
        full_text=""; used_model=ordered[0]; last_err=""
        for mid in ordered:
            if stop_ev.is_set(): break
            try:
                to = httpx.Timeout(connect=8.0, read=90.0, write=10.0, pool=5.0)
                async with httpx.AsyncClient(timeout=to) as client:
                    async with client.stream("POST",
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=hdrs,
                        json={"model":mid,"messages":messages,"temperature":0.2,"max_tokens":4000,"stream":True},
                    ) as resp:
                        if resp.status_code != 200:
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
        chat_sessions[sid].append({"role":"user","content":review_prompt})
        chat_sessions[sid].append({"role":"assistant","content":full_text})
        used_name=next((m["name"] for m in MODELS if m["id"]==used_model),"")
        yield f"data: {json.dumps({'done':True,'html':markdown_to_html(full_text),'used_model':used_name,'files':{},'raw':full_text})}\n\n"

    return StreamingResponse(streamer(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

# ── Project memory ─────────────────────────────────────────────────────────────
@app.post("/project/add")
async def project_add(files:List[UploadFile]=File(default=[]), sid:str=Form("default"),
                       leo_session:str|None=Cookie(default=None)):
    if r := require_auth(leo_session): return r
    project_memory.setdefault(sid,{})
    added=[]
    for f in files:
        if not f or not f.filename: continue
        try:
            raw=await f.read(); content=read_file(f.filename,raw,f.content_type or "")
            project_memory[sid][f.filename]=content; added.append(f.filename)
        except Exception as e: logger.error(f"Project {f.filename}: {e}")
    return JSONResponse({"ok":True,"added":added,"files":list(project_memory[sid].keys())})

@app.get("/project/list")
async def project_list(sid:str="default", leo_session:str|None=Cookie(default=None)):
    if r := require_auth(leo_session): return r
    mem=project_memory.get(sid,{})
    return JSONResponse({"files":[{"name":k,"size":len(v)} for k,v in mem.items()]})

@app.post("/project/remove")
async def project_remove(filename:str=Form(...), sid:str=Form("default"),
                          leo_session:str|None=Cookie(default=None)):
    if r := require_auth(leo_session): return r
    project_memory.get(sid,{}).pop(filename,None)
    return JSONResponse({"ok":True})

@app.post("/project/clear")
async def project_clear(sid:str=Form("default"), leo_session:str|None=Cookie(default=None)):
    if r := require_auth(leo_session): return r
    project_memory[sid]={}
    return JSONResponse({"ok":True})

# ── Chat stream ────────────────────────────────────────────────────────────────
@app.post("/chat-stream")
async def chat_stream(request:Request, prompt:str=Form(...), model_id:str=Form(None),
                       files:List[UploadFile]=File(default=[]), sid:str=Form("default"),
                       leo_session:str|None=Cookie(default=None)):
    if r := require_auth(leo_session): return r
    if not OPENROUTER_API_KEY:
        async def _e():
            yield f"data: {json.dumps({'error':'OPENROUTER_API_KEY not set'})}\n\n"
        return StreamingResponse(_e(), media_type="text/event-stream")

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
    chat_sessions.setdefault(sid,[])
    session=chat_sessions[sid]
    stop_ev=asyncio.Event(); stop_flags[sid]=stop_ev
    project_ctx=build_project_context(sid)
    sys_prompt=SYSTEM_PROMPT+("\n\n"+project_ctx if project_ctx else "")
    content_parts=[{"type":"text","text":user_text}]+img_parts
    msg_content=content_parts if (img_parts or len(content_parts)>1) else user_text
    messages=[{"role":"system","content":sys_prompt}]+session[-10:]+[{"role":"user","content":msg_content}]
    hdrs={"Authorization":f"Bearer {OPENROUTER_API_KEY}","Content-Type":"application/json",
          "HTTP-Referer":"https://leo-ai.app","X-Title":"Leo-AI"}

    async def streamer():
        full_text=""; used_model=ordered[0] if ordered else chosen; last_err=""
        for mid in ordered:
            if stop_ev.is_set(): break
            try:
                to=httpx.Timeout(connect=8.0,read=90.0,write=10.0,pool=5.0)
                async with httpx.AsyncClient(timeout=to) as client:
                    async with client.stream("POST",
                        "https://openrouter.ai/api/v1/chat/completions",headers=hdrs,
                        json={"model":mid,"messages":messages,"temperature":0.3,"max_tokens":4000,"stream":True},
                    ) as resp:
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
        session.append({"role":"user","content":prompt})
        session.append({"role":"assistant","content":full_text})
        files_found=extract_files(full_text)
        generated_files.setdefault(sid,{}).update(files_found)
        files_b64={n:base64.b64encode(c.encode()).decode() for n,c in files_found.items()}
        used_name=next((m["name"] for m in MODELS if m["id"]==used_model),"")
        yield f"data: {json.dumps({'done':True,'html':markdown_to_html(full_text),'files':files_b64,'used_model':used_name,'switched':switched,'raw':full_text})}\n\n"

    return StreamingResponse(streamer(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

# ── Other routes ───────────────────────────────────────────────────────────────
@app.post("/stop")
async def stop(sid:str=Form("default"), leo_session:str|None=Cookie(default=None)):
    if r := require_auth(leo_session): return r
    ev=stop_flags.get(sid)
    if ev: ev.set()
    return JSONResponse({"ok":True})

@app.post("/download")
async def download(filename:str=Form(...), code_b64:str=Form(...),
                    leo_session:str|None=Cookie(default=None)):
    if r := require_auth(leo_session): return r
    try: code=base64.b64decode(code_b64).decode("utf-8")
    except: return JSONResponse({"error":"invalid"}, status_code=400)
    ext=filename.rsplit(".",1)[-1].lower()
    mime={"py":"text/x-python","js":"application/javascript","ts":"text/typescript",
          "html":"text/html","css":"text/css","json":"application/json","txt":"text/plain",
          "md":"text/markdown","sh":"text/x-sh","go":"text/x-go","rs":"text/x-rust"}
    return Response(content=code, media_type=mime.get(ext,"text/plain"),
        headers={"Content-Disposition":f'attachment; filename="{filename}"',
                 "Content-Type":f'{mime.get(ext,"text/plain")}; charset=utf-8'})

@app.get("/download/zip/{sid}")
async def download_zip(sid:str="default", leo_session:str|None=Cookie(default=None)):
    if r := require_auth(leo_session): return r
    files=generated_files.get(sid,{})
    if not files: return JSONResponse({"error":"لا توجد ملفات"}, status_code=404)
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as zf:
        for fname,code in files.items(): zf.writestr(fname,code)
    buf.seek(0)
    return Response(content=buf.read(), media_type="application/zip",
        headers={"Content-Disposition":'attachment; filename="leo-ai.zip"'})

@app.post("/clear")
async def clear(sid:str=Form("default"), leo_session:str|None=Cookie(default=None)):
    if r := require_auth(leo_session): return r
    chat_sessions[sid]=[]; generated_files[sid]={}
    ev=stop_flags.get(sid)
    if ev: ev.set()
    return JSONResponse({"ok":True})

@app.post("/share")
async def share(html:str=Form(...), title:str=Form("محادثة"),
                 leo_session:str|None=Cookie(default=None)):
    if r := require_auth(leo_session): return r
    sid=hashlib.md5(f"{title}{time.time()}".encode()).hexdigest()[:8]
    shared_chats[sid]={"html":html,"title":title,"created":time.time()}
    return JSONResponse({"id":sid,"url":f"/share/{sid}"})

@app.get("/share/{sid}", response_class=HTMLResponse)
async def view_share(sid:str):
    # Shared chats are public (read-only, no sensitive data)
    chat=shared_chats.get(sid)
    if not chat: return HTMLResponse("<h2>غير موجود</h2>", status_code=404)
    return HTMLResponse(f"""<!DOCTYPE html><html lang="ar" dir="rtl">
<head><meta charset="UTF-8"><title>{chat['title']} — Leo-AI</title>
<style>body{{font-family:'Segoe UI',sans-serif;background:#212121;color:#ececec;padding:20px;max-width:800px;margin:0 auto}}
.cw2{{background:#1a1a1a;border-radius:9px;margin:8px 0;border:1px solid #404040;overflow:hidden}}
.ch{{display:flex;padding:6px 12px;background:#252525;border-bottom:1px solid #404040}}
.ch span{{color:#8e8ea0;font-size:11px;font-family:monospace}}pre{{margin:0;padding:12px;overflow-x:auto;font-size:12px;color:#cdd6f4}}
h1,h2,h3{{color:#19c37d;margin:10px 0 4px}}strong{{color:#fff}}
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
async def generate_image(prompt:str=Form(...), size:str=Form("1024x1024"), model:str=Form("flux"),
                          leo_session:str|None=Cookie(default=None)):
    if r := require_auth(leo_session): return r
    try:
        w,h=map(int,size.split("x")) if "x" in size else (1024,1024)
        return JSONResponse({"url":img_url(prompt,w,h,model),"prompt":prompt})
    except Exception as e:
        return JSONResponse({"error":str(e)}, status_code=500)

"""
Leo-AI v9 — Production Ready
- Admin: full access (GitHub, memory, alerts, all features)
- Guest: chat only
"""

import os, base64, logging, re, zipfile, io, json
import urllib.parse, asyncio, hashlib, time, secrets
import httpx

from fastapi import FastAPI, Request, Form, UploadFile, File, Cookie
from fastapi.responses import HTMLResponse, Response, JSONResponse, StreamingResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from typing import List

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("leo-ai")

app      = FastAPI(title="Leo-AI", version="9.0.0")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
GITHUB_TOKEN:       str = os.getenv("GITHUB_TOKEN", "")
APP_PASSWORD:       str = os.getenv("APP_PASSWORD", "")
AUTO_LOAD_REPO:     str = os.getenv("AUTO_LOAD_REPO", "")
ALPHA_VANTAGE_KEY:  str = os.getenv("ALPHA_VANTAGE_KEY", "demo")
NEWS_API_KEY:       str = os.getenv("NEWS_API_KEY", "")
ANTHROPIC_API_KEY:  str = os.getenv("ANTHROPIC_API_KEY", "")

# ── Session stores ─────────────────────────────────────────────────────────────
admin_sessions:  set[str]               = set()
guest_sessions:  set[str]               = set()
chat_sessions:   dict[str, list]        = {}
project_memory:  dict[str, dict]        = {}
stop_flags:      dict[str, asyncio.Event] = {}
generated_files: dict[str, dict]        = {}
shared_chats:    dict[str, dict]        = {}

# ── Auth ───────────────────────────────────────────────────────────────────────
def get_role(token):
    if not token: return "none"
    if token in admin_sessions: return "admin"
    if token in guest_sessions: return "guest"
    return "none"

def check_auth(token):
    if get_role(token) not in ("admin","guest"):
        return RedirectResponse(url="/login", status_code=302)
    return None

def check_admin(token):
    if get_role(token) != "admin":
        return JSONResponse({"error": "للمستخدم الرئيسي فقط"}, status_code=403)
    return None

def gh_headers():
    h = {"Accept": "application/vnd.github.v3+json", "User-Agent": "Leo-AI/9.0"}
    if GITHUB_TOKEN: h["Authorization"] = f"token {GITHUB_TOKEN}"
    return h

# ── Models ─────────────────────────────────────────────────────────────────────
CLAUDE_ID = "claude-sonnet-4-6"
# Claude فقط — الوحيد والرئيسي
MODELS = [
    {"id":CLAUDE_ID,"name":"Claude Sonnet 4.6","desc":"الأقوى • دائم التشغيل • 200K","badge":"👑","context":"200K","speed":"سريع","tag":"الرئيسي","vision":True,"strength":"code","provider":"anthropic"},
]
VISION_IDS = [CLAUDE_ID]
CODING_IDS = [CLAUDE_ID]

def smart_route(prompt: str) -> str:
    return CLAUDE_ID

# ── Free APIs ──────────────────────────────────────────────────────────────────
FREE_APIS = {
    "crypto":        {"name":"CoinGecko",        "url":"https://api.coingecko.com/api/v3",    "key_env":None,              "free_tier":"مجاني بدون مفتاح"},
    "currency":      {"name":"ExchangeRate-API", "url":"https://open.er-api.com/v6/latest",   "key_env":None,              "free_tier":"1500 طلب/شهر"},
    "weather":       {"name":"Open-Meteo",       "url":"https://api.open-meteo.com/v1",       "key_env":None,              "free_tier":"مجاني بدون مفتاح"},
    "stocks_global": {"name":"Alpha Vantage",    "url":"https://www.alphavantage.co/query",   "key_env":"ALPHA_VANTAGE_KEY","free_tier":"25 طلب/يوم"},
    "news":          {"name":"NewsAPI",           "url":"https://newsapi.org/v2",              "key_env":"NEWS_API_KEY",    "free_tier":"100 طلب/يوم"},
}

# ── Helpers ────────────────────────────────────────────────────────────────────
CODE_EXT = {".py",".js",".ts",".tsx",".jsx",".html",".css",".json",".md",".txt",
            ".yml",".yaml",".sh",".go",".rs",".java",".cpp",".c",".rb",".php",".vue",".sql"}
SKIP_DIRS = {"node_modules",".git","dist","build","__pycache__",".venv","venv","env",".next","coverage"}
MAX_FC = 15_000

def img_url(p, w=1024, h=1024, model="flux"):
    return f"https://image.pollinations.ai/prompt/{urllib.parse.quote(p)}?width={w}&height={h}&model={model}&nologo=true&seed={abs(hash(p))%99999}"

def read_file(fname, raw, ct):
    nl = fname.lower()
    if nl.endswith(".pdf") and HAS_PYPDF:
        try:
            reader = PdfReader(io.BytesIO(raw)); pages = []
            for i, page in enumerate(reader.pages[:50]):
                t = page.extract_text() or ""
                if t.strip(): pages.append(f"[صفحة {i+1}]\n{t.strip()}")
            return f"[PDF: {fname}]\n\n" + "\n\n".join(pages)[:MAX_FC]
        except Exception as e: return f"[PDF: {fname}] error: {e}"
    if nl.endswith(".zip"):
        parts = [f"[ZIP: {fname}]"]
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                for i in zf.infolist():
                    if i.is_dir() or i.file_size > 500_000: continue
                    try: parts.append(f"\n--- {i.filename} ---\n{zf.read(i.filename).decode('utf-8',errors='replace')[:3000]}")
                    except: pass
        except Exception as e: parts.append(f"(error: {e})")
        return "\n".join(parts)[:MAX_FC]
    try: return f"[{fname}]\n{raw.decode('utf-8',errors='replace')}"[:MAX_FC]
    except: return f"(cannot read: {fname})"

def build_project_ctx(sid):
    mem = project_memory.get(sid, {})
    if not mem: return ""
    parts = ["=== PROJECT CONTEXT ==="]
    for fname, content in mem.items():
        parts.append(f"\n### {fname}\n{content[:3000]}")
    parts.append("=== END PROJECT CONTEXT ===\n")
    return "\n".join(parts)

def markdown_to_html(text: str) -> str:
    code_blocks = []
    def extract(m):
        lang = m.group(1) or "text"; code = m.group(2); idx = len(code_blocks)
        code_blocks.append((lang, code)); return f"___CODE_{idx}___"
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
            p = p.replace("\n", "<br>")
            if not p.startswith("<"): p = f"<p>{p}</p>"
            result.append(p)
    text = "\n".join(result)
    for idx, (lang, code) in enumerate(code_blocks):
        esc_code = code.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        b64 = base64.b64encode(code.encode()).decode()
        html = (f'<div class="cw2"><div class="ch"><span>{lang}</span>'
                f'<div class="chbtns"><button onclick="cpy(this,\'{b64}\')">📋 نسخ</button>'
                f'<button onclick="explainCode(this,\'{b64}\')">💡 شرح</button></div></div>'
                f'<pre><code class="lang-{lang}">{esc_code}</code></pre></div>')
        text = text.replace(f"___CODE_{idx}___", html)
    return text

def extract_files(text):
    pat = re.compile(
        r"###\s*(\S+\.(?:py|html|css|js|json|txt|md|jsx|ts|tsx|vue|php|java|cpp|c|go|rs|rb|sh|sql|yaml|yml))\s*\n*```(?:\w+)?\n(.*?)```",
        re.DOTALL | re.IGNORECASE
    )
    files = {f.strip(): c.strip() for f, c in pat.findall(text)}
    if not files:
        m = re.search(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
        if m:
            c = m.group(1); cl = c.lower().strip()
            name = ("index.html" if cl.startswith(("<!doctype","<html"))
                    else "app.py" if "fastapi" in cl or cl.startswith("import ")
                    else "script.js" if "function " in cl
                    else "data.json" if cl.startswith(("{","["))
                    else "code.txt")
            files[name] = c.strip()
    return files

SYSTEM_PROMPT = """You are Leo-AI — a senior software engineer, UI/UX expert, and production-grade coding agent.

RULES:
- Write complete, working, production-quality code
- For HTML/CSS/JS: always write self-contained files with embedded CSS and JS
- Use modern design: dark themes, gradients, smooth animations, professional layouts
- Always include error handling, loading states, and responsive design
- When building dashboards: use Chart.js from CDN for charts
- When using APIs: always use fetch() with proper error handling
- Respond in the same language as the user (Arabic/English)
- Write code in markdown code blocks with language tags
- For multi-file projects: use ### filename.ext before each code block

IMPORTANT FOR PREVIEWS:
- Write complete HTML files starting with <!DOCTYPE html>
- Include ALL CSS and JS inside the HTML file
- Use CDN links for external libraries (Chart.js, etc.)
- Make sure the code runs standalone without any server
"""

# ══════════════════════════════════════════════════════════════════════════════
# LOGIN PAGE — Blue Dark Theme
# ══════════════════════════════════════════════════════════════════════════════
LOGIN_HTML = """<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<meta name="theme-color" content="#0a0f1e">
<title>Leo-AI</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box;-webkit-tap-highlight-color:transparent}}
:root{{--bg:#0a0f1e;--bg2:#0d1426;--sf:#111827;--bd:#1e2d4a;--bd2:#2a3f63;
      --ac:#b8c5d6;--ac2:#6b9fd4;--tx:#e8edf5;--mu:#6b7a99;--danger:#e85d6a}}
html,body{{height:100%;background:var(--bg);color:var(--tx);
  font-family:'Segoe UI',system-ui,sans-serif;
  display:flex;align-items:center;justify-content:center;padding:20px}}
.wrap{{width:100%;max-width:300px;display:flex;flex-direction:column;gap:10px}}
.logo-wrap{{text-align:center;margin-bottom:4px}}
.logo{{width:56px;height:56px;margin:0 auto 10px;
  background:linear-gradient(135deg,#1a2d4a,#0f1e38);
  border:1px solid var(--bd2);border-radius:14px;
  display:flex;align-items:center;justify-content:center;font-size:26px}}
h1{{font-size:20px;font-weight:700;color:var(--tx);text-align:center}}
.sub{{color:var(--mu);font-size:12px;margin-top:3px;text-align:center}}
.card{{background:var(--sf);border:1px solid var(--bd);border-radius:12px;padding:16px}}
.ctitle{{font-size:11px;font-weight:700;margin-bottom:11px;display:flex;
  align-items:center;gap:6px;color:var(--mu);letter-spacing:.5px;text-transform:uppercase}}
.badge{{font-size:9px;font-weight:700;border-radius:4px;padding:2px 6px}}
.badge.adm{{background:rgba(184,197,214,.1);color:var(--ac);border:1px solid rgba(184,197,214,.2)}}
.badge.gst{{background:rgba(107,159,212,.1);color:var(--ac2);border:1px solid rgba(107,159,212,.2)}}
input[type=password]{{width:100%;background:var(--bg);color:var(--tx);
  border:1px solid var(--bd);border-radius:9px;padding:11px 13px;font-size:15px;
  outline:none;text-align:center;letter-spacing:3px;margin-bottom:10px;
  font-family:inherit;transition:border-color .15s}}
input[type=password]:focus{{border-color:var(--bd2)}}
input[type=password]::placeholder{{letter-spacing:1px;font-size:13px}}
.btn{{width:100%;padding:11px;border:none;border-radius:9px;font-size:14px;
  font-weight:700;cursor:pointer;transition:opacity .15s}}
.btn:active{{opacity:.85}}
.btn-admin{{background:linear-gradient(135deg,#b8c5d6,#8fa3bf);color:#0a0f1e}}
.btn-guest{{background:transparent;color:var(--ac2);border:1px solid rgba(107,159,212,.3)}}
.btn-guest:active{{background:rgba(107,159,212,.08)}}
.err{{color:var(--danger);font-size:11.5px;text-align:center;min-height:16px;margin-top:4px}}
.div{{display:flex;align-items:center;gap:10px;color:var(--mu);font-size:11px}}
.div::before,.div::after{{content:'';flex:1;height:1px;background:var(--bd)}}
</style>
</head>
<body>
<div class="wrap">
  <div class="logo-wrap">
    <div class="logo">🤖</div>
    <h1>Leo-AI</h1>
    <p class="sub">مساعد البرمجة الاحترافي</p>
  </div>
  <div class="card">
    <div class="ctitle">🔐 المستخدم الرئيسي <span class="badge adm">ADMIN</span></div>
    <form method="post" action="/login/admin">
      <input type="password" name="password" placeholder="كلمة المرور" autocomplete="current-password" autofocus>
      <button class="btn btn-admin" type="submit">دخول ←</button>
    </form>
    <div class="err">{admin_err}</div>
  </div>
  <div class="div">أو</div>
  <div class="card">
    <div class="ctitle">👤 دخول كضيف <span class="badge gst">GUEST</span></div>
    <form method="post" action="/login/guest">
      <button class="btn btn-guest" type="submit">دخول بدون كلمة مرور ←</button>
    </form>
  </div>
</div>
</body>
</html>"""

# ══════════════════════════════════════════════════════════════════════════════
# AUTH ROUTES
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/login", response_class=HTMLResponse)
async def login_page(err: str = ""):
    admin_err = "❌ كلمة المرور خاطئة" if err == "1" else ""
    return HTMLResponse(LOGIN_HTML.format(admin_err=admin_err))

@app.post("/login/admin")
async def login_admin(password: str = Form(...)):
    if APP_PASSWORD and password != APP_PASSWORD:
        return RedirectResponse(url="/login?err=1", status_code=302)
    token = secrets.token_urlsafe(32)
    admin_sessions.add(token)
    resp = RedirectResponse(url="/", status_code=302)
    resp.set_cookie("leo_session", token, max_age=30*24*3600, httponly=True, samesite="lax")
    resp.set_cookie("leo_role", "admin", max_age=30*24*3600, samesite="lax")
    return resp

@app.post("/login/guest")
async def login_guest():
    token = secrets.token_urlsafe(32)
    guest_sessions.add(token)
    resp = RedirectResponse(url="/", status_code=302)
    resp.set_cookie("leo_session", token, max_age=30*24*3600, httponly=True, samesite="lax")
    resp.set_cookie("leo_role", "guest", max_age=30*24*3600, samesite="lax")
    return resp

@app.get("/logout")
async def logout_get(leo_session: str | None = Cookie(default=None)):
    admin_sessions.discard(leo_session or "")
    guest_sessions.discard(leo_session or "")
    resp = RedirectResponse(url="/login", status_code=302)
    resp.delete_cookie("leo_session"); resp.delete_cookie("leo_role")
    return resp

@app.post("/logout")
async def logout_post(leo_session: str | None = Cookie(default=None)):
    admin_sessions.discard(leo_session or "")
    guest_sessions.discard(leo_session or "")
    resp = RedirectResponse(url="/login", status_code=302)
    resp.delete_cookie("leo_session"); resp.delete_cookie("leo_role")
    return resp

# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse)
async def home(request: Request, leo_session: str | None = Cookie(default=None)):
    if r := check_auth(leo_session): return r
    role = get_role(leo_session)
    # Auto-load repo for admin
    if role == "admin" and AUTO_LOAD_REPO and "/" in AUTO_LOAD_REPO:
        sid = "default"
        if sid not in project_memory or not project_memory[sid]:
            try:
                owner, repo = AUTO_LOAD_REPO.split("/", 1)
                async with httpx.AsyncClient(timeout=20, headers=gh_headers()) as client:
                    tree_resp = await client.get(f"https://api.github.com/repos/{owner}/{repo}/git/trees/HEAD?recursive=1")
                    if tree_resp.status_code == 200:
                        project_memory.setdefault(sid, {})
                        for item in tree_resp.json().get("tree", [])[:30]:
                            if item["type"] != "blob": continue
                            ext = os.path.splitext(item["path"])[1].lower()
                            if ext not in CODE_EXT: continue
                            fr = await client.get(f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/{item['path']}")
                            if fr.status_code == 200:
                                project_memory[sid][f"{repo}/{item['path']}"] = fr.text[:5000]
            except Exception as e:
                logger.warning(f"Auto-load failed: {e}")
    return templates.TemplateResponse("index.html", {
        "request":   request,
        "models":    MODELS,
        "role":      role,
        "is_admin":  role == "admin",
    })

# ══════════════════════════════════════════════════════════════════════════════
# SMART ROUTE INFO
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/smart-route")
async def smart_route_info(prompt: str = "", leo_session: str | None = Cookie(default=None)):
    if r := check_auth(leo_session): return r
    mid = smart_route(prompt) if prompt else MODELS[0]["id"]
    m = next((x for x in MODELS if x["id"] == mid), MODELS[0])
    return JSONResponse({"model_id": mid, "model_name": m["name"], "badge": m["badge"]})

# ══════════════════════════════════════════════════════════════════════════════
# FREE APIs
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/apis")
async def get_apis(leo_session: str | None = Cookie(default=None)):
    if r := check_auth(leo_session): return r
    result = {}
    for key, api in FREE_APIS.items():
        result[key] = {
            "name": api["name"],
            "url": api["url"],
            "free_tier": api["free_tier"],
            "configured": bool(api["key_env"] is None or os.getenv(api["key_env"] or "", "")),
        }
    return JSONResponse(result)

@app.get("/proxy/crypto")
async def proxy_crypto(coin: str = "bitcoin,ethereum,solana", leo_session: str | None = Cookie(default=None)):
    if r := check_auth(leo_session): return r
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get("https://api.coingecko.com/api/v3/simple/price",
                params={"ids": coin, "vs_currencies": "usd,sar", "include_24hr_change": "true"})
            return JSONResponse(resp.json())
    except Exception as e: return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/proxy/currency")
async def proxy_currency(base: str = "USD", leo_session: str | None = Cookie(default=None)):
    if r := check_auth(leo_session): return r
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"https://open.er-api.com/v6/latest/{base}")
            return JSONResponse(resp.json())
    except Exception as e: return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/proxy/weather")
async def proxy_weather(lat: float = 24.7, lon: float = 46.7, leo_session: str | None = Cookie(default=None)):
    if r := check_auth(leo_session): return r
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get("https://api.open-meteo.com/v1/forecast",
                params={"latitude": lat, "longitude": lon,
                        "current": "temperature_2m,wind_speed_10m,precipitation,weathercode",
                        "daily": "temperature_2m_max,temperature_2m_min,weathercode",
                        "forecast_days": 7, "timezone": "auto"})
            return JSONResponse(resp.json())
    except Exception as e: return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/proxy/stock")
async def proxy_stock(symbol: str = "AAPL", leo_session: str | None = Cookie(default=None)):
    if r := check_auth(leo_session): return r
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get("https://www.alphavantage.co/query",
                params={"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": ALPHA_VANTAGE_KEY})
            return JSONResponse(resp.json())
    except Exception as e: return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/proxy/news")
async def proxy_news(q: str = "technology", leo_session: str | None = Cookie(default=None)):
    if r := check_auth(leo_session): return r
    if not NEWS_API_KEY:
        return JSONResponse({"message": "أضف NEWS_API_KEY في Render", "articles": []})
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get("https://newsapi.org/v2/everything",
                params={"q": q, "language": "ar", "pageSize": 10, "apiKey": NEWS_API_KEY})
            return JSONResponse(resp.json())
    except Exception as e: return JSONResponse({"error": str(e)}, status_code=500)

# ══════════════════════════════════════════════════════════════════════════════
# ADMIN ALERTS
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/admin/alerts")
async def admin_alerts(leo_session: str | None = Cookie(default=None)):
    if r := check_admin(leo_session): return r
    alerts = []

    # 1. OpenRouter API Key
    if not OPENROUTER_API_KEY:
        alerts.append({"level":"error","icon":"🔑","title":"OPENROUTER_API_KEY مفقود",
            "detail":"البوت لن يعمل بدونه. أضفه في Render > Environment Variables"})
    else:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r2 = await client.get("https://openrouter.ai/api/v1/models",
                    headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"})
                if r2.status_code == 401:
                    alerts.append({"level":"error","icon":"🔑","title":"OPENROUTER_API_KEY غير صالح",
                        "detail":"المفتاح مرفوض. تحقق منه أو جدده في openrouter.ai"})
                else:
                    alerts.append({"level":"ok","icon":"✅","title":"OpenRouter API يعمل",
                        "detail":"المفتاح صالح والاتصال ناجح"})
        except Exception as e:
            alerts.append({"level":"warn","icon":"🌐","title":"تعذر الاتصال بـ OpenRouter",
                "detail":f"تحقق من الإنترنت. ({str(e)[:60]})"})

    # 2. GitHub Token
    if not GITHUB_TOKEN:
        alerts.append({"level":"warn","icon":"🐙","title":"GITHUB_TOKEN مفقود",
            "detail":"لن تتمكن من تصفح repos. أضف GITHUB_TOKEN في Render"})
    else:
        try:
            async with httpx.AsyncClient(timeout=8, headers=gh_headers()) as client:
                r3 = await client.get("https://api.github.com/user")
                if r3.status_code == 200:
                    uname = r3.json().get("login", "")
                    alerts.append({"level":"ok","icon":"🐙","title":f"GitHub متصل — @{uname}",
                        "detail":"التوكن صالح ويمكن تصفح المستودعات"})
                else:
                    alerts.append({"level":"error","icon":"🐙","title":"GITHUB_TOKEN منتهي أو خاطئ",
                        "detail":"جدد التوكن من GitHub > Settings > Developer settings > Personal access tokens"})
        except Exception as e:
            alerts.append({"level":"warn","icon":"🐙","title":"تعذر الاتصال بـ GitHub","detail":str(e)[:80]})

    # 3. Test each model
    if OPENROUTER_API_KEY:
        hdrs = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
        test_msg = [{"role": "user", "content": "hi"}]
        broken = []; working = []
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=20.0, write=5.0, pool=5.0)) as client:
            for m in MODELS:
                try:
                    resp = await client.post("https://openrouter.ai/api/v1/chat/completions",
                        headers=hdrs, json={"model": m["id"], "messages": test_msg, "max_tokens": 3, "stream": False})
                    if resp.status_code == 200:
                        working.append(f"{m['badge']} {m['name']}")
                    else:
                        try: err_msg = resp.json().get("error", {}).get("message", "")[:100]
                        except: err_msg = f"HTTP {resp.status_code}"
                        if "No endpoints" in err_msg or "not found" in err_msg.lower():
                            reason = "أُزيل من OpenRouter أو غير متاح مجاناً"
                            fix = "أبلغ المطور لاستبداله في app.py"
                        elif "rate" in err_msg.lower() or "quota" in err_msg.lower():
                            reason = "تجاوزت حد الاستخدام اليومي"
                            fix = "انتظر إعادة التعيين (يومياً) أو أضف رصيداً"
                        else:
                            reason = err_msg or f"HTTP {resp.status_code}"
                            fix = "حاول مرة أخرى أو اختر موديلاً آخر"
                        broken.append(f"• {m['badge']} {m['name']}: {reason}\n  → الحل: {fix}")
                except Exception as exc:
                    broken.append(f"• {m['badge']} {m['name']}: خطأ اتصال — {str(exc)[:60]}")

        if working:
            alerts.append({"level":"ok","icon":"🤖","title":f"{len(working)} موديل يعمل",
                "detail":" • ".join(working)})
        if broken:
            alerts.append({"level":"error","icon":"🤖","title":f"{len(broken)} موديل متوقف",
                "detail":"\n".join(broken)})

    # 4. Password check
    if not APP_PASSWORD:
        alerts.append({"level":"warn","icon":"🔒","title":"APP_PASSWORD غير مضبوط",
            "detail":"أي شخص يمكنه الدخول كـ Admin. أضف APP_PASSWORD في Render"})

    return JSONResponse({
        "alerts": alerts,
        "errors":   sum(1 for a in alerts if a["level"] == "error"),
        "warnings": sum(1 for a in alerts if a["level"] == "warn"),
        "ok":       sum(1 for a in alerts if a["level"] == "ok"),
    })

# ══════════════════════════════════════════════════════════════════════════════
# GITHUB
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/github/repos")
async def github_repos(leo_session: str | None = Cookie(default=None)):
    if r := check_admin(leo_session): return r
    try:
        async with httpx.AsyncClient(timeout=15, headers=gh_headers()) as client:
            resp = await client.get("https://api.github.com/user/repos",
                params={"sort":"updated","per_page":30,"type":"all"})
            if resp.status_code != 200:
                return JSONResponse({"error": f"GitHub: {resp.status_code}"}, status_code=resp.status_code)
            return JSONResponse([{"name":r["name"],"full_name":r["full_name"],
                "private":r["private"],"language":r.get("language") or "—",
                "updated_at":r["updated_at"][:10],"stars":r["stargazers_count"],
                "default_branch":r.get("default_branch","main")} for r in resp.json()])
    except Exception as e: return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/github/tree")
async def github_tree(owner: str, repo: str, branch: str = "main", leo_session: str | None = Cookie(default=None)):
    if r := check_admin(leo_session): return r
    try:
        async with httpx.AsyncClient(timeout=20, headers=gh_headers()) as client:
            repo_resp = await client.get(f"https://api.github.com/repos/{owner}/{repo}")
            if repo_resp.status_code == 404:
                return JSONResponse({"error": "الـ repo غير موجود"}, status_code=404)
            repo_data = repo_resp.json(); branch = repo_data.get("default_branch", branch)
            tree_resp = await client.get(f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1")
            if tree_resp.status_code != 200:
                return JSONResponse({"error": "فشل جلب الشجرة"}, status_code=500)
            files = []
            for item in tree_resp.json().get("tree", []):
                if item["type"] != "blob": continue
                path = item["path"]
                if any(p in SKIP_DIRS for p in path.split("/")): continue
                ext = os.path.splitext(path)[1].lower()
                files.append({"path":path,"size":item.get("size",0),"is_code":ext in CODE_EXT})
            return JSONResponse({"repo":f"{owner}/{repo}","branch":branch,"files":files})
    except Exception as e: return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/github/file")
async def github_file(owner: str, repo: str, path: str, branch: str = "main", leo_session: str | None = Cookie(default=None)):
    if r := check_admin(leo_session): return r
    try:
        async with httpx.AsyncClient(timeout=15, headers=gh_headers()) as client:
            resp = await client.get(f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}")
            if resp.status_code == 404: return JSONResponse({"error": "الملف غير موجود"}, status_code=404)
            return JSONResponse({"path":path,"content":resp.text[:20_000]})
    except Exception as e: return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/github/load-to-memory")
async def github_load(owner: str = Form(...), repo: str = Form(...), branch: str = Form("main"),
                       paths: str = Form(""), sid: str = Form("default"), leo_session: str | None = Cookie(default=None)):
    if r := check_admin(leo_session): return r
    path_list = [p.strip() for p in paths.split(",") if p.strip()][:30]
    project_memory.setdefault(sid, {}); added = []
    async with httpx.AsyncClient(timeout=20, headers=gh_headers()) as client:
        for path in path_list:
            resp = await client.get(f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}")
            if resp.status_code == 200:
                project_memory[sid][f"{repo}/{path}"] = f"[GitHub: {owner}/{repo}/{path}]\n{resp.text[:10_000]}"
                added.append(path)
    return JSONResponse({"ok":True,"added":added,"files":list(project_memory[sid].keys())})

@app.post("/github/load-repo-full")
async def github_load_full(owner: str = Form(...), repo: str = Form(...), branch: str = Form("main"),
                            sid: str = Form("default"), leo_session: str | None = Cookie(default=None)):
    if r := check_admin(leo_session): return r
    try:
        project_memory.setdefault(sid, {}); added = 0
        async with httpx.AsyncClient(timeout=30, headers=gh_headers()) as client:
            tree_resp = await client.get(f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1")
            if tree_resp.status_code != 200: return JSONResponse({"error":"فشل جلب الشجرة"},status_code=500)
            code_files = [i for i in tree_resp.json().get("tree",[])
                          if i["type"]=="blob" and os.path.splitext(i["path"])[1].lower() in CODE_EXT
                          and not any(p in SKIP_DIRS for p in i["path"].split("/"))
                          and i.get("size",0) < 50_000][:30]
            for item in code_files:
                fr = await client.get(f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{item['path']}")
                if fr.status_code == 200:
                    project_memory[sid][f"{repo}/{item['path']}"] = fr.text[:8000]
                    added += 1
        return JSONResponse({"ok":True,"added":added,"repo":repo,"files":list(project_memory[sid].keys())})
    except Exception as e: return JSONResponse({"error":str(e)},status_code=500)

@app.post("/github/review")
async def github_review(owner: str = Form(...), repo: str = Form(...), branch: str = Form("main"),
                         paths: str = Form(""), model_id: str = Form(None), sid: str = Form("default"),
                         leo_session: str | None = Cookie(default=None)):
    if r := check_admin(leo_session): return r
    if not OPENROUTER_API_KEY:
        async def _e():
            yield f"data: {json.dumps({'error':'API key not set'})}\n\n"
        return StreamingResponse(_e(), media_type="text/event-stream")

    path_list = [p.strip() for p in paths.split(",") if p.strip()][:20]
    file_contents = {}
    async with httpx.AsyncClient(timeout=20, headers=gh_headers()) as client:
        for path in path_list:
            resp = await client.get(f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}")
            if resp.status_code == 200: file_contents[path] = resp.text[:8_000]

    if not file_contents:
        async def _e():
            yield f"data: {json.dumps({'error':'لم يتم جلب أي ملف'})}\n\n"
        return StreamingResponse(_e(), media_type="text/event-stream")

    files_block = "\n\n".join(f"### {p}\n```\n{c}\n```" for p, c in file_contents.items())
    review_prompt = f"راجع هذا الكود من {owner}/{repo} وأعطني:\n1. تقييم /10\n2. الأخطاء\n3. مشاكل الأمان\n4. مشاكل الأداء\n5. الكود المصحح\n\n{files_block}"

    valid_ids = [m["id"] for m in MODELS]
    chosen = model_id if model_id in valid_ids else MODELS[0]["id"]
    ordered = [chosen] + [mid for mid in CODING_IDS if mid != chosen]
    messages = [{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":review_prompt}]
    hdrs = {"Authorization":f"Bearer {OPENROUTER_API_KEY}","Content-Type":"application/json","HTTP-Referer":"https://leo-ai.app","X-Title":"Leo-AI"}

    async def streamer():
        full_text = ""; last_err = ""; used_model = ordered[0]
        for mid in ordered:
            try:
                to = httpx.Timeout(connect=8.0, read=90.0, write=10.0, pool=5.0)
                async with httpx.AsyncClient(timeout=to) as client:
                    async with client.stream("POST","https://openrouter.ai/api/v1/chat/completions",
                        headers=hdrs, json={"model":mid,"messages":messages,"temperature":0.2,"max_tokens":6000,"stream":True}) as resp:
                        if resp.status_code != 200:
                            body = await resp.aread()
                            try: last_err = json.loads(body).get("error",{}).get("message","")
                            except: last_err = body.decode(errors="replace")[:200]
                            continue
                        used_model = mid; buf = ""; got_first = False
                        async for chunk in resp.aiter_text():
                            buf += chunk; lines = buf.split("\n"); buf = lines.pop()
                            for line in lines:
                                line = line.strip()
                                if not line.startswith("data:"): continue
                                ds = line[5:].strip()
                                if ds == "[DONE]": break
                                try:
                                    delta = json.loads(ds)["choices"][0]["delta"].get("content","")
                                    if delta:
                                        if not got_first: got_first=True; yield f"data: {json.dumps({'first':True})}\n\n"
                                        full_text += delta; yield f"data: {json.dumps({'token':delta})}\n\n"
                                except: continue
                break
            except: continue
        if not full_text: full_text = f"⚠️ فشل: {last_err}"; yield f"data: {json.dumps({'token':full_text})}\n\n"
        used_name = next((m["name"] for m in MODELS if m["id"]==used_model),"")
        yield f"data: {json.dumps({'done':True,'html':markdown_to_html(full_text),'used_model':used_name,'files':{},'raw':full_text})}\n\n"

    return StreamingResponse(streamer(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.post("/github/push")
async def github_push(owner: str = Form(...), repo: str = Form(...), branch: str = Form("main"),
                       files: str = Form(...), message: str = Form("Leo-AI: update files"),
                       leo_session: str | None = Cookie(default=None)):
    if r := check_admin(leo_session): return r
    if not GITHUB_TOKEN: return JSONResponse({"error":"GITHUB_TOKEN مفقود"},status_code=401)
    try: files_dict = json.loads(files)
    except: return JSONResponse({"error":"تنسيق خاطئ"},status_code=400)

    results = []; errors = []
    hdrs2 = {**gh_headers(), "Content-Type":"application/json"}
    async with httpx.AsyncClient(timeout=25, headers=hdrs2) as client:
        repo_resp = await client.get(f"https://api.github.com/repos/{owner}/{repo}")
        if repo_resp.status_code != 200: return JSONResponse({"error":"الـ repo غير موجود"},status_code=404)
        branch = repo_resp.json().get("default_branch", branch)
        for filepath, b64_content in files_dict.items():
            try:
                check = await client.get(f"https://api.github.com/repos/{owner}/{repo}/contents/{filepath}",params={"ref":branch})
                sha = check.json().get("sha") if check.status_code == 200 else None
                payload = {"message":message,"content":b64_content,"branch":branch}
                if sha: payload["sha"] = sha
                resp = await client.put(f"https://api.github.com/repos/{owner}/{repo}/contents/{filepath}",json=payload)
                if resp.status_code in (200,201): results.append(filepath)
                else: errors.append(f"{filepath}: HTTP {resp.status_code}")
            except Exception as exc: errors.append(f"{filepath}: {exc}")

    return JSONResponse({"ok":len(errors)==0,"pushed":results,"errors":errors,"repo":f"{owner}/{repo}","branch":branch})

# ══════════════════════════════════════════════════════════════════════════════
# PROJECT MEMORY
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/project/add")
async def project_add(files: List[UploadFile] = File(default=[]), sid: str = Form("default"),
                       leo_session: str | None = Cookie(default=None)):
    if r := check_admin(leo_session): return r
    project_memory.setdefault(sid, {}); added = []
    for f in files:
        if not f or not f.filename: continue
        try:
            raw = await f.read()
            project_memory[sid][f.filename] = read_file(f.filename, raw, f.content_type or "")
            added.append(f.filename)
        except Exception as e: logger.error(f"Project {f.filename}: {e}")
    return JSONResponse({"ok":True,"added":added,"files":list(project_memory[sid].keys())})

@app.post("/project/clear")
async def project_clear(sid: str = Form("default"), leo_session: str | None = Cookie(default=None)):
    if r := check_admin(leo_session): return r
    project_memory[sid] = {}; return JSONResponse({"ok":True})

@app.get("/project/list")
async def project_list(sid: str = "default", leo_session: str | None = Cookie(default=None)):
    if r := check_admin(leo_session): return r
    mem = project_memory.get(sid, {})
    return JSONResponse({"files":[{"name":k,"size":len(v)} for k,v in mem.items()]})

# ══════════════════════════════════════════════════════════════════════════════
# SCREENSHOT TO CODE
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/screenshot-to-code")
async def screenshot_to_code(image: UploadFile = File(...), style: str = Form("dark"),
                              framework: str = Form("html"), leo_session: str | None = Cookie(default=None)):
    if r := check_auth(leo_session): return r
    if not OPENROUTER_API_KEY:
        async def _e():
            yield f"data: {json.dumps({'error':'API key not set'})}\n\n"
        return StreamingResponse(_e(), media_type="text/event-stream")

    try:
        raw = await image.read(); mt = image.content_type or "image/png"
        b64_img = base64.b64encode(raw).decode()
    except Exception as exc:
        async def _e():
            yield f"data: {json.dumps({'error':str(exc)})}\n\n"
        return StreamingResponse(_e(), media_type="text/event-stream")

    style_map = {
        "dark":    "dark theme with #0a0f1e background, silver/blue accents, professional",
        "modern":  "modern clean white design with subtle shadows",
        "tailwind": "using Tailwind-like CSS",
    }
    style_desc = style_map.get(style, style_map["dark"])
    fw_desc = "React JSX component" if framework == "react" else "single self-contained HTML file with embedded CSS and JS"

    prompt = (f"Recreate this UI screenshot as {fw_desc}.\n"
              f"Style: {style_desc}\n"
              "Requirements: pixel-perfect layout, responsive, all visible elements, "
              "realistic placeholder data, hover effects. Output ONLY the complete code.")

    messages = [
        {"role":"system","content":"You are an expert UI developer. Recreate UIs from screenshots accurately."},
        {"role":"user","content":[
            {"type":"text","text":prompt},
            {"type":"image_url","image_url":{"url":f"data:{mt};base64,{b64_img}"}},
        ]},
    ]
    hdrs = {"Authorization":f"Bearer {OPENROUTER_API_KEY}","Content-Type":"application/json","HTTP-Referer":"https://leo-ai.app","X-Title":"Leo-AI"}

    async def streamer():
        full_text = ""; last_err = ""
        for mid in VISION_IDS:
            try:
                to = httpx.Timeout(connect=8.0, read=90.0, write=10.0, pool=5.0)
                async with httpx.AsyncClient(timeout=to) as client:
                    async with client.stream("POST","https://openrouter.ai/api/v1/chat/completions",
                        headers=hdrs, json={"model":mid,"messages":messages,"temperature":0.1,"max_tokens":4000,"stream":True}) as resp:
                        if resp.status_code != 200:
                            body = await resp.aread()
                            try: last_err = json.loads(body).get("error",{}).get("message","")
                            except: last_err = body.decode(errors="replace")[:200]
                            continue
                        buf = ""; got_first = False
                        async for chunk in resp.aiter_text():
                            buf += chunk; lines = buf.split("\n"); buf = lines.pop()
                            for line in lines:
                                line = line.strip()
                                if not line.startswith("data:"): continue
                                ds = line[5:].strip()
                                if ds == "[DONE]": break
                                try:
                                    delta = json.loads(ds)["choices"][0]["delta"].get("content","")
                                    if delta:
                                        if not got_first: got_first=True; yield f"data: {json.dumps({'first':True,'model':mid})}\n\n"
                                        full_text += delta; yield f"data: {json.dumps({'token':delta})}\n\n"
                                except: continue
                break
            except Exception as exc: last_err = str(exc); continue

        if not full_text:
            full_text = f"فشل التحويل: {last_err}"
            yield f"data: {json.dumps({'token':full_text})}\n\n"

        code_match = re.search(r"```(?:html|jsx|tsx)?\n([\s\S]*?)```", full_text)
        code = code_match.group(1).strip() if code_match else full_text.strip()
        fname = "component.jsx" if framework == "react" else "screenshot.html"
        files_b64 = {fname: base64.b64encode(code.encode()).decode()}
        yield f"data: {json.dumps({'done':True,'html':markdown_to_html(full_text),'files':files_b64,'raw':full_text,'used_model':'Vision Model'})}\n\n"

    return StreamingResponse(streamer(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

# ══════════════════════════════════════════════════════════════════════════════
# CHAT STREAM
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/chat-stream")
async def chat_stream(request: Request, prompt: str = Form(...), model_id: str = Form(None),
                       files: List[UploadFile] = File(default=[]), sid: str = Form("default"),
                       leo_session: str | None = Cookie(default=None)):
    if r := check_auth(leo_session): return r
    if not ANTHROPIC_API_KEY:
        async def _no_key():
            yield "data: " + json.dumps({"error": "ANTHROPIC_API_KEY غير موجود في Render — أضفه في Environment Variables"}) + "\n\n"
        return StreamingResponse(_no_key(), media_type="text/event-stream")

    role = get_role(leo_session)
    if role == "guest":
        sid = f"guest_{sid}"

    user_text = prompt
    img_parts = []
    has_image = False

    real_files = [f for f in (files or []) if f and f.filename][:25]
    for f in real_files:
        try:
            raw = await f.read()
            mt = f.content_type or ""
            if mt.startswith("image/"):
                has_image = True
                b64 = base64.b64encode(raw).decode()
                img_parts.append({"type": "image", "source": {"type": "base64", "media_type": mt, "data": b64}})
            else:
                user_text += f"\n\n{read_file(f.filename, raw, mt)}"
        except Exception as e:
            logger.error(f"File {f.filename}: {e}")

    chat_sessions.setdefault(sid, [])
    session = chat_sessions[sid]
    stop_ev = asyncio.Event()
    stop_flags[sid] = stop_ev

    sys_prompt = SYSTEM_PROMPT
    if role == "admin":
        ctx = build_project_ctx(sid)
        if ctx:
            sys_prompt += "\n\n" + ctx

    # بناء رسائل Anthropic
    an_messages = []
    for m in session[-12:]:
        an_messages.append({"role": m["role"], "content": m["content"]})

    # الرسالة الحالية
    if img_parts:
        user_content = [{"type": "text", "text": user_text}] + img_parts
    else:
        user_content = user_text
    an_messages.append({"role": "user", "content": user_content})

    hdrs = {
        "x-api-key": ANTHROPIC_API_KEY,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 8000,
        "system": sys_prompt,
        "messages": an_messages,
        "stream": True,
    }

    async def streamer():
        full_text = ""
        last_err = ""
        got_first = False
        try:
            rt = 180.0 if any(w in prompt.lower() for w in ["مشروع كامل", "full project", "منصة كاملة"]) else 90.0
            to = httpx.Timeout(connect=8.0, read=rt, write=10.0, pool=5.0)
            async with httpx.AsyncClient(timeout=to) as client:
                async with client.stream("POST", "https://api.anthropic.com/v1/messages",
                                         headers=hdrs, json=body) as resp:
                    if resp.status_code != 200:
                        err_body = await resp.aread()
                        try:
                            last_err = json.loads(err_body).get("error", {}).get("message", "")
                        except:
                            last_err = err_body.decode(errors="replace")[:300]
                        full_text = f"❌ خطأ من Claude API: {last_err}"
                        yield "data: " + json.dumps({"token": full_text}) + "\n\n"
                    else:
                        buf = ""
                        async for chunk in resp.aiter_text():
                            if stop_ev.is_set():
                                break
                            buf += chunk
                            events = buf.split("\n")
                            buf = events.pop()
                            for line in events:
                                line = line.strip()
                                if not line.startswith("data:"):
                                    continue
                                ds = line[5:].strip()
                                try:
                                    ev = json.loads(ds)
                                    if ev.get("type") == "content_block_delta":
                                        delta = ev.get("delta", {}).get("text", "")
                                        if delta:
                                            if not got_first:
                                                got_first = True
                                                yield "data: " + json.dumps({"first": True}) + "\n\n"
                                            full_text += delta
                                            yield "data: " + json.dumps({"token": delta}) + "\n\n"
                                except:
                                    continue
        except httpx.ReadTimeout:
            full_text = "⏱️ **انتهت مهلة الانتظار**\n\n- اضغط 🔄 وأرسل مرة أخرى\n- قسّم الطلب لأجزاء أصغر"
            yield "data: " + json.dumps({"token": full_text}) + "\n\n"
        except Exception as exc:
            full_text = f"⚠️ **خطأ في الاتصال**\n\n`{str(exc)[:200]}`\n\n- اضغط 🔄 إعادة الاتصال"
            yield "data: " + json.dumps({"token": full_text}) + "\n\n"

        if stop_ev.is_set() and full_text:
            full_text += "\n\n*(⏹ توقف)*"

        session.append({"role": "user", "content": prompt})
        session.append({"role": "assistant", "content": full_text})
        files_found = extract_files(full_text)
        generated_files.setdefault(sid, {}).update(files_found)
        files_b64 = {n: base64.b64encode(c.encode()).decode() for n, c in files_found.items()}
        yield "data: " + json.dumps({
            "done": True,
            "html": markdown_to_html(full_text),
            "files": files_b64,
            "used_model": "Claude Sonnet 4.6",
            "switched": False,
            "raw": full_text,
            "role": role,
        }) + "\n\n"

    return StreamingResponse(streamer(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/stop")
async def stop(sid: str = Form("default"), leo_session: str | None = Cookie(default=None)):
    if r := check_auth(leo_session): return r
    role = get_role(leo_session)
    if role == "guest": sid = f"guest_{sid}"
    ev = stop_flags.get(sid)
    if ev: ev.set()
    return JSONResponse({"ok":True})

@app.post("/download")
async def download(filename: str = Form(...), code_b64: str = Form(...), leo_session: str | None = Cookie(default=None)):
    if r := check_auth(leo_session): return r
    try: code = base64.b64decode(code_b64).decode("utf-8")
    except: return JSONResponse({"error":"invalid"}, status_code=400)
    ext = filename.rsplit(".",1)[-1].lower()
    mime = {"py":"text/x-python","js":"application/javascript","html":"text/html","css":"text/css","json":"application/json","txt":"text/plain","md":"text/markdown"}
    return Response(content=code, media_type=mime.get(ext,"text/plain"),
        headers={"Content-Disposition":f'attachment; filename="{filename}"'})

@app.get("/download/zip/{sid}")
async def download_zip(sid: str = "default", leo_session: str | None = Cookie(default=None)):
    if r := check_auth(leo_session): return r
    files = generated_files.get(sid, {})
    if not files: return JSONResponse({"error":"لا توجد ملفات"}, status_code=404)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, code in files.items(): zf.writestr(fname, code)
    buf.seek(0)
    return Response(content=buf.read(), media_type="application/zip",
        headers={"Content-Disposition":'attachment; filename="leo-ai.zip"'})

@app.post("/clear")
async def clear(sid: str = Form("default"), leo_session: str | None = Cookie(default=None)):
    if r := check_auth(leo_session): return r
    role = get_role(leo_session)
    if role == "guest": sid = f"guest_{sid}"
    chat_sessions[sid] = []; generated_files[sid] = {}
    ev = stop_flags.get(sid)
    if ev: ev.set()
    return JSONResponse({"ok":True})

@app.post("/share")
async def share(html: str = Form(...), title: str = Form("محادثة"), leo_session: str | None = Cookie(default=None)):
    if r := check_auth(leo_session): return r
    sid = hashlib.md5(f"{title}{time.time()}".encode()).hexdigest()[:8]
    shared_chats[sid] = {"html":html,"title":title,"created":time.time()}
    return JSONResponse({"id":sid,"url":f"/share/{sid}"})

@app.get("/share/{sid}", response_class=HTMLResponse)
async def view_share(sid: str):
    chat = shared_chats.get(sid)
    if not chat: return HTMLResponse("<h2>غير موجود</h2>", status_code=404)
    return HTMLResponse(f"""<!DOCTYPE html><html lang="ar" dir="rtl">
<head><meta charset="UTF-8"><title>{chat['title']}</title>
<style>body{{font-family:'Segoe UI',sans-serif;background:#0a0f1e;color:#e8edf5;padding:20px;max-width:800px;margin:0 auto}}</style>
</head><body><h2 style="color:#b8c5d6;margin-bottom:16px">🤖 {chat['title']}</h2>{chat['html']}</body></html>""")

@app.post("/generate-image")
async def generate_image(prompt: str = Form(...), size: str = Form("1024x1024"), model: str = Form("flux"),
                          leo_session: str | None = Cookie(default=None)):
    if r := check_auth(leo_session): return r
    try:
        w, h = map(int, size.split("x")) if "x" in size else (1024, 1024)
        return JSONResponse({"url":img_url(prompt, w, h, model),"prompt":prompt})
    except Exception as e: return JSONResponse({"error":str(e)}, status_code=500)

@app.get("/health")
async def health():
    return JSONResponse({
        "status": "ok",
        "version": "9.1.0-claude-only",
        "models": len(MODELS),
        "model_names": [m["name"] for m in MODELS],
        "claude_ready": bool(ANTHROPIC_API_KEY),
    })

import os
import base64
import requests
import logging
import re
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, Response, JSONResponse
from fastapi.templating import Jinja2Templates

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

chat_sessions = {}
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

FALLBACK_MODELS = [
    "google/gemma-4-26b-a4b-it:free",
    "openai/gpt-oss-120b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "poolside/laguna-m.1:free",
    "meta-llama/llama-3.3-70b-instruct:free",
]

SYSTEM_PROMPT = (
    "أنت مساعد برمجي ذكي ومتفاعل. قواعدك:\n"
    "1. تفاعل مع المستخدم بالمحادثة - اسأل لتوضيح المتطلبات\n"
    "2. اشرح فهمك قبل كتابة الكود\n"
    "3. اكتب الأكواد في بلوكات Markdown واضحة\n"
    "4. لو المستخدم يحتاج أكثر من ملف، اكتب كل ملف منفصل مع اسم الملف\n"
    "5. استخدم هذا التنسيق لكل ملف:\n"
    "   ### filename.ext\n"
    "   ```language\n"
    "   // code here\n"
    "   ```\n"
    "6. كن سريعاً ومختصراً في الشرح\n"
    "7. اكتب بالعربية دائماً"
)

def extract_files_from_response(text):
    """Extract multiple files from AI response with ### filename.ext pattern"""
    files = {}

    # Pattern: ### filename.ext followed by code block
    pattern = r'###\s*(\S+\.(?:py|html|css|js|json|txt|md|jsx|ts|tsx|vue|php|java|cpp|c|go|rs|swift|kt|dart|rb|pl|sh|sql|xml|yaml|yml))\s*\n*```(?:\w+)?\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

    for filename, code in matches:
        files[filename.strip()] = code.strip()

    # If no files found with ### pattern, try single code block
    if not files:
        code_pattern = r'```(?:\w+)?\n(.*?)```'
        code_matches = re.findall(code_pattern, text, re.DOTALL)
        if code_matches:
            content = code_matches[0]
            filename = guess_filename(content)
            files[filename] = content.strip()

    return files

def guess_filename(content):
    """Guess filename from code content"""
    content_lower = content.lower().strip()

    if content_lower.startswith('<!doctype html>') or content_lower.startswith('<html'):
        return 'index.html'
    elif 'fastapi' in content_lower or 'flask' in content_lower or content_lower.startswith('import '):
        if 'def ' in content or 'class ' in content:
            return 'app.py'
    elif content_lower.startswith('const ') or content_lower.startswith('let ') or content_lower.startswith('var ') or 'function ' in content_lower:
        return 'script.js'
    elif content_lower.startswith('body {') or content_lower.startswith('* {') or '.class' in content_lower or '#id' in content_lower:
        return 'style.css'
    elif content_lower.startswith('{') or content_lower.startswith('['):
        return 'data.json'

    return 'generated_code.txt'

def get_file_icon(filename):
    """Get emoji icon for file type"""
    ext = filename.split('.')[-1].lower()
    icons = {
        'py': '🐍', 'js': '📜', 'html': '🌐', 'css': '🎨',
        'json': '📋', 'txt': '📄', 'md': '📝', 'jsx': '⚛️',
        'ts': '📘', 'tsx': '📘', 'vue': '🟢', 'php': '🐘',
        'java': '☕', 'cpp': '⚙️', 'c': '⚙️', 'go': '🐹',
        'rs': '🦀', 'swift': '🐦', 'kt': '🟣', 'dart': '🎯',
        'rb': '💎', 'sql': '🗄️', 'xml': '📰', 'yaml': '⚙️', 'yml': '⚙️'
    }
    return icons.get(ext, '📄')

def get_language_label(filename):
    """Get human-readable language name"""
    ext = filename.split('.')[-1].lower()
    labels = {
        'py': 'Python', 'js': 'JavaScript', 'html': 'HTML',
        'css': 'CSS', 'json': 'JSON', 'txt': 'Text',
        'md': 'Markdown', 'jsx': 'React JSX', 'ts': 'TypeScript',
        'tsx': 'React TSX', 'vue': 'Vue.js', 'php': 'PHP',
        'java': 'Java', 'cpp': 'C++', 'c': 'C', 'go': 'Go',
        'rs': 'Rust', 'swift': 'Swift', 'kt': 'Kotlin',
        'dart': 'Dart', 'rb': 'Ruby', 'sql': 'SQL',
        'xml': 'XML', 'yaml': 'YAML', 'yml': 'YAML'
    }
    return labels.get(ext, 'Code')

@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "chat_history": [],
        "files": {},
        "user_prompt": ""
    })

@app.post("/", response_class=HTMLResponse)
async def chat_with_ai(
    request: Request,
    prompt: str = Form(...),
    file: UploadFile = File(None),
    session_id: str = Form("default")
):
    logger.info(f"Chat: {prompt[:50]}...")

    if not OPENROUTER_API_KEY:
        error_msg = "⚠️ خطأ: لم يتم إعداد مفتاح API. اذهب إلى إعدادات Render وأضف OPENROUTER_API_KEY"
        return templates.TemplateResponse("index.html", {
            "request": request,
            "chat_history": [{"role": "user", "content": prompt}, {"role": "ai", "content": error_msg}],
            "files": {},
            "user_prompt": ""
        })

    if session_id not in chat_sessions:
        chat_sessions[session_id] = []

    session = chat_sessions[session_id]

    content_list = [{"type": "text", "text": prompt}]

    if file and file.filename != "":
        try:
            file_content = await file.read()
            file_mime = file.content_type

            if file_mime.startswith("image/"):
                base64_image = base64.b64encode(file_content).decode("utf-8")
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{file_mime};base64,{base64_image}"}
                })
            else:
                try:
                    text_data = file_content.decode("utf-8")
                    content_list[0]["text"] += f"\n\n[محتوى الملف المرفق {file.filename}]:\n{text_data}"
                except:
                    content_list[0]["text"] += f"\n\n(تم إرفاق ملف غير نصي: {file.filename})"
        except Exception as e:
            logger.error(f"File error: {e}")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in session[-6:]:
        messages.append(msg)
    messages.append({"role": "user", "content": content_list})

    ai_response = ""
    last_error = ""

    for model_name in FALLBACK_MODELS:
        try:
            logger.info(f"Trying: {model_name}")
            data = {
                "model": model_name,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 4000
            }

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=20
            )

            logger.info(f"Status: {response.status_code}")
            response_json = response.json()

            if "choices" in response_json and response_json["choices"]:
                ai_response = response_json["choices"][0]["message"]["content"]
                logger.info(f"Success: {model_name}")
                break
            elif "error" in response_json:
                last_error = f"{model_name}: {response_json['error'].get('message', 'خطأ')}"
                logger.warning(last_error)
                continue

        except requests.exceptions.Timeout:
            last_error = f"{model_name}: انتهى الوقت"
            continue
        except Exception as e:
            last_error = f"{model_name}: {str(e)}"
            continue

    if not ai_response:
        ai_response = f"❌ تم رفض الطلب من جميع النماذج. آخر خطأ: {last_error}"

    # Update session
    session.append({"role": "user", "content": prompt})
    session.append({"role": "assistant", "content": ai_response})

    # Extract files
    files = extract_files_from_response(ai_response)

    # Build chat history for display (convert markdown to HTML)
    chat_history = []
    for i in range(0, len(session), 2):
        if i < len(session):
            user_msg = session[i]["content"] if isinstance(session[i]["content"], str) else str(session[i]["content"])
            chat_history.append({"role": "user", "content": user_msg})
        if i + 1 < len(session):
            ai_msg = session[i + 1]["content"]
            # Convert markdown code blocks to HTML for display
            ai_msg_html = ai_msg.replace("```", "<pre><code>").replace("```", "</code></pre>")
            chat_history.append({"role": "ai", "content": ai_msg_html})

    return templates.TemplateResponse("index.html", {
        "request": request,
        "chat_history": chat_history,
        "files": files,
        "user_prompt": ""
    })

@app.post("/download")
async def download_file(filename: str = Form(...), code_content: str = Form(...)):
    ext = filename.split('.')[-1].lower()
    mime_types = {
        'py': 'text/x-python', 'js': 'application/javascript',
        'html': 'text/html', 'css': 'text/css',
        'json': 'application/json', 'txt': 'text/plain',
        'md': 'text/markdown', 'jsx': 'text/javascript',
        'ts': 'text/typescript', 'tsx': 'text/typescript',
        'vue': 'text/javascript', 'php': 'text/php',
        'java': 'text/x-java', 'cpp': 'text/x-c++src',
        'c': 'text/x-csrc', 'go': 'text/x-go',
        'rs': 'text/x-rust', 'swift': 'text/x-swift',
        'kt': 'text/x-kotlin', 'dart': 'text/x-dart',
        'rb': 'text/x-ruby', 'sql': 'text/x-sql',
        'xml': 'text/xml', 'yaml': 'text/yaml', 'yml': 'text/yaml'
    }

    return Response(
        content=code_content,
        media_type=mime_types.get(ext, 'text/plain'),
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.post("/clear")
async def clear_chat(session_id: str = Form("default")):
    if session_id in chat_sessions:
        chat_sessions[session_id] = []
    return JSONResponse({"status": "cleared"})

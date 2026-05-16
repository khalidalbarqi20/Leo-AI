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
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

# Fastest models first - 5 second timeout
FALLBACK_MODELS = [
    "google/gemma-4-26b-a4b-it:free",
    "openai/gpt-oss-120b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "poolside/laguna-m.1:free",
    "meta-llama/llama-3.3-70b-instruct:free",
]

SYSTEM_PROMPT = (
    "You are a fast coding assistant. Rules:\n"
    "1. Be extremely concise - max 2 sentences explanation\n"
    "2. Write code immediately in Markdown blocks\n"
    "3. For multiple files use: ### filename.ext followed by code block\n"
    "4. Always respond in Arabic"
)

def markdown_to_html(text):
    """Fast markdown to HTML"""
    code_blocks = []
    def extract_code(match):
        lang = match.group(1) or 'text'
        code = match.group(2)
        idx = len(code_blocks)
        code_blocks.append((lang, code))
        return f"___CODE_{idx}___"

    text = re.sub(r'```(\w+)?\n(.*?)```', extract_code, text, flags=re.DOTALL)
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    text = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    text = re.sub(r'^> (.*?)$', r'<blockquote>\1</blockquote>', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\.\s+(.*?)$', r'<li>\1</li>', text, flags=re.MULTILINE)

    paragraphs = text.split('\n\n')
    result = []
    for p in paragraphs:
        p = p.strip()
        if p:
            p = p.replace('\n', '<br>')
            if not p.startswith('<'):
                p = f'<p>{p}</p>'
            result.append(p)
    text = '\n'.join(result)

    for idx, (lang, code) in enumerate(code_blocks):
        escaped = code.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        b64 = base64.b64encode(code.encode('utf-8')).decode('utf-8')
        display = lang if lang else 'code'
        html = (
            '<div class="code-wrap">'
            '<div class="code-hdr">'
            '<span>' + display + '</span>'
            '<button onclick="cpy(this,\'' + b64 + '\')">&#128203; نسخ</button>'
            '</div><pre><code>' + escaped + '</code></pre></div>'
        )
        text = text.replace(f"___CODE_{idx}___", html)
    return text

def extract_files(text):
    """Extract files from response"""
    files = {}
    pattern = r'###\s*(\S+\.(?:py|html|css|js|json|txt|md|jsx|ts|tsx|vue|php|java|cpp|c|go|rs|swift|kt|dart|rb|sh|sql|xml|yaml|yml))\s*\n*```(?:\w+)?\n(.*?)```'
    for fname, code in re.findall(pattern, text, re.DOTALL | re.IGNORECASE):
        files[fname.strip()] = code.strip()
    if not files:
        m = re.search(r'```(?:\w+)?\n(.*?)```', text, re.DOTALL)
        if m:
            c = m.group(1)
            files[guess_name(c)] = c.strip()
    return files

def guess_name(content):
    """Guess filename"""
    cl = content.lower().strip()
    if cl.startswith('<!doctype') or cl.startswith('<html'):
        return 'index.html'
    elif 'fastapi' in cl or 'flask' in cl or cl.startswith('import '):
        return 'app.py'
    elif cl.startswith('const ') or cl.startswith('let ') or 'function ' in cl:
        return 'script.js'
    elif 'body {' in cl or '.class' in cl:
        return 'style.css'
    elif cl.startswith('{') or cl.startswith('['):
        return 'data.json'
    return 'code.txt'

@app.get('/', response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse('index.html', {
        'request': request,
        'chat_history': [],
        'files': {},
        'user_prompt': ''
    })

@app.post('/', response_class=HTMLResponse)
async def chat(request: Request, prompt: str = Form(...), file: UploadFile = File(None)):
    if not OPENROUTER_API_KEY:
        return templates.TemplateResponse('index.html', {
            'request': request,
            'chat_history': [{'role': 'user', 'content': prompt}, {'role': 'ai', 'content': 'Error: API key not set'}],
            'files': {}, 'user_prompt': ''})

    sid = 'default'
    if sid not in chat_sessions:
        chat_sessions[sid] = []
    session = chat_sessions[sid]

    content = [{'type': 'text', 'text': prompt}]
    if file and file.filename:
        try:
            fc = await file.read()
            mt = file.content_type
            if mt.startswith('image/'):
                b64 = base64.b64encode(fc).decode('utf-8')
                content.append({'type': 'image_url', 'image_url': {'url': f'data:{mt};base64,{b64}'}})
            else:
                try:
                    td = fc.decode('utf-8')
                    content[0]['text'] += f'\n\n[File {file.filename}]:\n{td}'
                except:
                    content[0]['text'] += f'\n\n(Non-text: {file.filename})'
        except Exception as e:
            logger.error(f'File: {e}')

    headers = {'Authorization': f'Bearer {OPENROUTER_API_KEY}', 'Content-Type': 'application/json'}
    msgs = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    msgs.extend(session[-5:])
    msgs.append({'role': 'user', 'content': content})

    ai_resp = ''
    last_err = ''
    for model in FALLBACK_MODELS:
        try:
            r = requests.post('https://openrouter.ai/api/v1/chat/completions',
                headers=headers, json={'model': model, 'messages': msgs, 'temperature': 0.7, 'max_tokens': 2000}, timeout=8)
            j = r.json()
            if 'choices' in j and j['choices']:
                ai_resp = j['choices'][0]['message']['content']
                break
            elif 'error' in j:
                last_err = j['error'].get('message', 'error')
                continue
        except:
            continue

    if not ai_resp:
        ai_resp = f'All models failed. Error: {last_err}'

    session.append({'role': 'user', 'content': prompt})
    session.append({'role': 'assistant', 'content': ai_resp})

    files = extract_files(ai_resp)
    history = []
    for i in range(0, len(session), 2):
        if i < len(session):
            um = session[i]['content'] if isinstance(session[i]['content'], str) else str(session[i]['content'])
            history.append({'role': 'user', 'content': um})
        if i + 1 < len(session):
            history.append({'role': 'ai', 'content': markdown_to_html(session[i+1]['content'])})

    return templates.TemplateResponse('index.html', {
        'request': request, 'chat_history': history, 'files': files, 'user_prompt': ''})

@app.post('/download')
async def download(filename: str = Form(...), code_content: str = Form(...)):
    ext = filename.split('.')[-1].lower()
    mime = {'py': 'text/x-python', 'js': 'application/javascript', 'html': 'text/html',
        'css': 'text/css', 'json': 'application/json', 'txt': 'text/plain',
        'md': 'text/markdown', 'jsx': 'text/javascript', 'ts': 'text/typescript',
        'tsx': 'text/typescript', 'vue': 'text/javascript', 'php': 'text/php',
        'java': 'text/x-java', 'cpp': 'text/x-c++src', 'c': 'text/x-csrc',
        'go': 'text/x-go', 'rs': 'text/x-rust', 'swift': 'text/x-swift',
        'kt': 'text/x-kotlin', 'dart': 'text/x-dart', 'rb': 'text/x-ruby',
        'sql': 'text/x-sql', 'xml': 'text/xml', 'yaml': 'text/yaml', 'yml': 'text/yaml'}
    return Response(content=code_content, media_type=mime.get(ext, 'text/plain'),
        headers={'Content-Disposition': f'attachment; filename={filename}'}

@app.post('/clear')
async def clear():
    chat_sessions['default'] = []
    return JSONResponse({'ok': True})

import os
import base64
import requests
import logging
import re
import json
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

FALLBACK_MODELS = [
    "google/gemma-4-26b-a4b-it:free",
    "openai/gpt-oss-120b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "poolside/laguna-m.1:free",
    "meta-llama/llama-3.3-70b-instruct:free",
]

SYSTEM_PROMPT = (
    "You are a helpful coding assistant. Rules:\n"
    "1. Interact conversationally - ask clarifying questions\n"
    "2. Explain your understanding before writing code\n"
    "3. Write code in clear Markdown blocks\n"
    "4. If user needs multiple files, write each separately with filename\n"
    "5. Use this format for each file:\n"
    "   ### filename.ext\n"
    "   ```language\n"
    "   // code here\n"
    "   ```\n"
    "6. Be quick and concise in explanations\n"
    "7. Always respond in Arabic"
)

def markdown_to_html(text):
    """Convert markdown to HTML with copy buttons for code blocks"""
    code_blocks = []

    def extract_code_block(match):
        lang = match.group(1) or 'text'
        code = match.group(2)
        idx = len(code_blocks)
        code_blocks.append((lang, code))
        return f"__CODE_BLOCK_{idx}__"

    text = re.sub(r'```(\w+)?\n(.*?)```', extract_code_block, text, flags=re.DOTALL)

    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    text = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*\*(.*?)\*\*\*', r'<strong><em>\1</em></strong>', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    text = re.sub(r'^> (.*?)$', r'<blockquote>\1</blockquote>', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\.\s+(.*?)$', r'<li>\1</li>', text, flags=re.MULTILINE)

    paragraphs = text.split('\n\n')
    processed_paragraphs = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        p = p.replace('\n', '<br>')
        if not p.startswith('<'):
            p = f'<p>{p}</p>'
        processed_paragraphs.append(p)

    text = '\n'.join(processed_paragraphs)

    lang_map = {
        'python': 'python', 'py': 'python',
        'javascript': 'javascript', 'js': 'javascript',
        'typescript': 'typescript', 'ts': 'typescript',
        'html': 'markup', 'xml': 'markup',
        'css': 'css', 'json': 'json',
        'bash': 'bash', 'sh': 'bash',
        'jsx': 'jsx', 'tsx': 'tsx',
        'java': 'java', 'cpp': 'cpp', 'c': 'c',
        'go': 'go', 'rust': 'rust', 'rs': 'rust',
        'php': 'php', 'ruby': 'ruby', 'rb': 'ruby',
        'sql': 'sql', 'yaml': 'yaml', 'yml': 'yaml'
    }

    for idx, (lang, code) in enumerate(code_blocks):
        prism_lang = lang_map.get(lang.lower(), 'text')
        display_lang = lang if lang else 'code'
        escaped_code = code.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        code_b64 = base64.b64encode(code.encode('utf-8')).decode('utf-8')

        code_html = (
            '<div class="code-block-wrapper">\n'
            '    <div class="code-block-header">\n'
            '        <span class="code-lang">' + display_lang + '</span>\n'
            '        <button class="btn-copy-code" onclick="copyCode(this, \'' + code_b64 + '\')">&#128203; نسخ</button>\n'
            '    </div>\n'
            '    <pre><code class="language-' + prism_lang + '">' + escaped_code + '</code></pre>\n'
            '</div>'
        )

        text = text.replace(f"__CODE_BLOCK_{idx}__", code_html)

    return text

def extract_files_from_response(text):
    """Extract multiple files from AI response"""
    files = {}
    pattern = r'###\s*(\S+\.(?:py|html|css|js|json|txt|md|jsx|ts|tsx|vue|php|java|cpp|c|go|rs|swift|kt|dart|rb|pl|sh|sql|xml|yaml|yml))\s*\n*```(?:\w+)?\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    for filename, code in matches:
        files[filename.strip()] = code.strip()
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

@app.get('/', response_class=HTMLResponse)
async def home_page(request: Request):
    return templates.TemplateResponse('index.html', {
        'request': request,
        'chat_history': [],
        'files': {},
        'user_prompt': ''
    })

@app.post('/', response_class=HTMLResponse)
async def chat_with_ai(
    request: Request,
    prompt: str = Form(...),
    file: UploadFile = File(None),
    session_id: str = Form('default')
):
    logger.info(f'Chat: {prompt[:50]}...')
    if not OPENROUTER_API_KEY:
        error_msg = 'Error: OPENROUTER_API_KEY not set'
        return templates.TemplateResponse('index.html', {
            'request': request,
            'chat_history': [{'role': 'user', 'content': prompt}, {'role': 'ai', 'content': error_msg}],
            'files': {},
            'user_prompt': ''
        })

    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    session = chat_sessions[session_id]

    content_list = [{'type': 'text', 'text': prompt}]
    if file and file.filename != '':
        try:
            file_content = await file.read()
            file_mime = file.content_type
            if file_mime.startswith('image/'):
                base64_image = base64.b64encode(file_content).decode('utf-8')
                content_list.append({
                    'type': 'image_url',
                    'image_url': {'url': f'data:{file_mime};base64,{base64_image}'}
                })
            else:
                try:
                    text_data = file_content.decode('utf-8')
                    content_list[0]['text'] += f'\n\n[File {file.filename}]:\n{text_data}'
                except:
                    content_list[0]['text'] += f'\n\n(Non-text file: {file.filename})'
        except Exception as e:
            logger.error(f'File error: {e}')

    headers = {
        'Authorization': f'Bearer {OPENROUTER_API_KEY}',
        'Content-Type': 'application/json'
    }

    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    for msg in session[-10:]:
        messages.append(msg)
    messages.append({'role': 'user', 'content': content_list})

    ai_response = ''
    last_error = ''
    for model_name in FALLBACK_MODELS:
        try:
            logger.info(f'Trying: {model_name}')
            data = {
                'model': model_name,
                'messages': messages,
                'temperature': 0.7,
                'max_tokens': 4000
            }
            response = requests.post(
                'https://openrouter.ai/api/v1/chat/completions',
                headers=headers,
                json=data,
                timeout=20
            )
            logger.info(f'Status: {response.status_code}')
            response_json = response.json()
            if 'choices' in response_json and response_json['choices']:
                ai_response = response_json['choices'][0]['message']['content']
                logger.info(f'Success: {model_name}')
                break
            elif 'error' in response_json:
                last_error = f'{model_name}: {response_json["error"].get("message", "error")}'
                logger.warning(last_error)
                continue
        except requests.exceptions.Timeout:
            last_error = f'{model_name}: timeout'
            continue
        except Exception as e:
            last_error = f'{model_name}: {str(e)}'
            continue

    if not ai_response:
        ai_response = f'All models failed. Last error: {last_error}'

    session.append({'role': 'user', 'content': prompt})
    session.append({'role': 'assistant', 'content': ai_response})

    files = extract_files_from_response(ai_response)

    chat_history = []
    for i in range(0, len(session), 2):
        if i < len(session):
            user_msg = session[i]['content'] if isinstance(session[i]['content'], str) else str(session[i]['content'])
            chat_history.append({'role': 'user', 'content': user_msg})
        if i + 1 < len(session):
            ai_msg_raw = session[i + 1]['content']
            ai_msg_html = markdown_to_html(ai_msg_raw)
            chat_history.append({'role': 'ai', 'content': ai_msg_html})

    return templates.TemplateResponse('index.html', {
        'request': request,
        'chat_history': chat_history,
        'files': files,
        'user_prompt': ''
    })

@app.post('/download')
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
        headers={'Content-Disposition': f'attachment; filename={filename}'}
    )

@app.post('/clear')
async def clear_chat(session_id: str = Form('default')):
    if session_id in chat_sessions:
        chat_sessions[session_id] = []
    return JSONResponse({'status': 'cleared'})

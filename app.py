import os
import base64
import requests
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = "anthropic/claude-sonnet-4.6"

@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "response": "", "raw_code": ""})

@app.post("/", response_class=HTMLResponse)
async def generate_code_and_analyze(
    request: Request, 
    prompt: str = Form(...), 
    file: UploadFile = File(None)
):
    if not OPENROUTER_API_KEY:
        return templates.TemplateResponse("index.html", {"request": request, "response": "خطأ: لم يتم ضبط الـ API Key", "raw_code": ""})

    # تجهيز محتوى الرسالة الافتراضي للـ API
    content_list = [{"type": "text", "text": prompt}]

    # معالجة الملف أو الصورة المرفوعة إن وجدت
    if file and file.filename != "":
        file_content = await file.read()
        file_mime = file.content_type
        
        # إذا كان المرفق صورة، نقوم بتشفيرها وإرسالها للتحليل البصري
        if file_mime.startswith("image/"):
            base64_image = base64.b64encode(file_content).decode("utf-8")
            content_list.append({
                "type": "image_url",
                "image_url": {"url": f"data:{file_mime};base64,{base64_image}"}
            })
        else:
            # إذا كان ملف نصي أو كود برمي، نقرأ محتواه كمتن نصي مدمج
            try:
                text_data = file_content.decode("utf-8")
                content_list[0]["text"] += f"\n\n[محتوى الملف المرفق {file.filename}]:\n{text_data}"
            except:
                content_list[0]["text"] += f"\n\n(تم إرفاق ملف غير نصي: {file.filename})"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    system_instruction = (
        "أنت محطة تطوير متكاملة وخبير برمجيات أقدم. "
        "مهمتك هي تحليل الطلبات والملفات المرفقة (أكواد أو صور تصاميم)، وكتابة حلول برمجية احترافية. "
        "يجب أن تفصل الشرح العربي عن الكود البرمجي البرمجي الحقيقي لكي يتمكن المستخدم من تحميله مباشرة."
    )

    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": content_list}
        ]
    }

    ai_response = ""
    raw_code = ""
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        response_json = response.json()
        ai_response = response_json['choices'][0]['message']['content']
        
        # استخراج الكود النظيف فقط لغرض ميزة التحميل كملف جاهز
        if "```" in ai_response:
            parts = ai_response.split("```")
            # أخذ أول كود برمجي يظهر في الإجابة وتجريده من اسم اللغة
            raw_code = parts[1].split("\n", 1)[1] if "\n" in parts[1] else parts[1]
        else:
            raw_code = ai_response
    except Exception as e:
        ai_response = f"حدث خطأ في الاتصال بالسيرفر السحابي: {str(e)}"

    return templates.TemplateResponse("index.html", {
        "request": request, 
        "response": ai_response, 
        "user_prompt": prompt,
        "raw_code": raw_code
    })

# مسار جديد تماماً مخصص لتحميل الكود الناتج بملف مستقل جاهز للتنزيل
@app.post("/download")
async def download_file(code_content: str = Form(...)):
    filename = "generated_code.txt"
    # محاولة ذكية لتخمين نوع الملف وحفظه بالصيغة المناسبة
    if "import " in code_content or "def " in code_content: filename = "script.py"
    elif "<html" in code_content.lower(): filename = "index.html"
    elif "const " in code_content or "let " in code_content: filename = "script.js"
    
    return Response(
        content=code_content,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

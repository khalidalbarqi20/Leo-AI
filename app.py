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

# ✅ قائمة نماذج احتياطية — يجربهم بالترتيب تلقائياً
FALLBACK_MODELS = [
    "google/gemma-4-26b-a4b-it:free",      # Google — مستقر
    "openai/gpt-oss-120b:free",             # OpenAI — قوي
    "nvidia/nemotron-3-super-120b-a12b:free", # NVIDIA — سريع
    "poolside/laguna-m.1:free",             # مخصص للأكواد
    "meta-llama/llama-3.3-70b-instruct:free", # Llama — احتياطي
]

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
        error_msg = "خطأ أمني: لم يتم العثور على مفتاح الـ API في إعدادات Render."
        return templates.TemplateResponse("index.html", {"request": request, "response": error_msg, "raw_code": ""})

    content_list = [{"type": "text", "text": prompt}]

    if file and file.filename != "":
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

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    system_instruction = (
        "أنت مساعد برمجيات ذكي خبير جداً ومخصص للمستخدم. "
        "مهمتك الأساسية هي تلقي المتطلبات باللغة العربية، وكتابة كود برمجي كامل ونظيف "
        "داخل بلوكات برمجية واضحة (Markdown Code Blocks) مع توفير شرح مبسط ومباشر باللغة العربية."
    )

    ai_response = ""
    raw_code = ""
    last_error = ""

    # ✅ جرب كل نموذج بالترتيب حتى يشتغل واحد
    for model_name in FALLBACK_MODELS:
        try:
            data = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": content_list}
                ]
            }

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            response_json = response.json()

            if 'choices' in response_json and response_json['choices']:
                ai_response = response_json['choices'][0]['message']['content']

                if "```" in ai_response:
                    parts = ai_response.split("```")
                    raw_code = parts[1].split("\n", 1)[1] if "\n" in parts[1] else parts[1]
                else:
                    raw_code = ai_response

                # ✅ اشتغل — اخرج من الحلقة
                break

            elif 'error' in response_json:
                last_error = f"{model_name}: {response_json['error'].get('message', 'غير معروف')}"
                continue  # جرب النموذج التالي

        except Exception as e:
            last_error = f"{model_name}: {str(e)}"
            continue  # جرب النموذج التالي

    # لو ما اشتغل ولا نموذج
    if not ai_response:
        ai_response = f"تم رفض الطلب من جميع النماذج المتاحة. آخر خطأ: {last_error}"

    return templates.TemplateResponse("index.html", {
        "request": request, 
        "response": ai_response, 
        "user_prompt": prompt,
        "raw_code": raw_code
    })

@app.post("/download")
async def download_file(code_content: str = Form(...)):
    filename = "generated_code.txt"
    if "import " in code_content or "def " in code_content: filename = "script.py"
    elif "<html" in code_content.lower(): filename = "index.html"
    elif "const " in code_content or "let " in code_content: filename = "script.js"

    return Response(
        content=code_content,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

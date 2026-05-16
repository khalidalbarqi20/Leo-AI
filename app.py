import os
import base64
import requests
import logging
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Fallback models - tries each until one works
FALLBACK_MODELS = [
    "google/gemma-4-26b-a4b-it:free",
    "openai/gpt-oss-120b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "poolside/laguna-m.1:free",
    "meta-llama/llama-3.3-70b-instruct:free",
]

@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "response": "",
        "raw_code": "",
        "user_prompt": ""
    })

@app.post("/", response_class=HTMLResponse)
async def generate_code_and_analyze(
    request: Request,
    prompt: str = Form(...),
    file: UploadFile = File(None)
):
    logger.info(f"Received prompt: {prompt[:50]}...")

    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY not found")
        error_msg = "خطأ أمني: لم يتم العثور على مفتاح الـ API في إعدادات Render."
        return templates.TemplateResponse("index.html", {
            "request": request,
            "response": error_msg,
            "raw_code": "",
            "user_prompt": prompt
        })

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
            logger.error(f"File processing error: {e}")

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

    for model_name in FALLBACK_MODELS:
        try:
            logger.info(f"Trying model: {model_name}")
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
                timeout=15  # Reduced timeout to prevent Render from killing
            )

            logger.info(f"Response status: {response.status_code}")
            response_json = response.json()

            if "choices" in response_json and response_json["choices"]:
                ai_response = response_json["choices"][0]["message"]["content"]
                logger.info(f"Success with model: {model_name}")

                if "```" in ai_response:
                    parts = ai_response.split("```")
                    if len(parts) > 1:
                        raw_code = parts[1].split("\n", 1)[1] if "\n" in parts[1] else parts[1]
                else:
                    raw_code = ai_response

                break

            elif "error" in response_json:
                last_error = f"{model_name}: {response_json['error'].get('message', 'غير معروف')}"
                logger.warning(last_error)
                continue

        except requests.exceptions.Timeout:
            last_error = f"{model_name}: انتهى الوقت (Timeout)"
            logger.warning(last_error)
            continue
        except Exception as e:
            last_error = f"{model_name}: {str(e)}"
            logger.error(last_error)
            continue

    if not ai_response:
        ai_response = f"تم رفض الطلب من جميع النماذج المتاحة. آخر خطأ: {last_error}"
        logger.error("All models failed")

    return templates.TemplateResponse("index.html", {
        "request": request,
        "response": ai_response,
        "user_prompt": prompt,
        "raw_code": raw_code
    })

@app.post("/download")
async def download_file(code_content: str = Form(...)):
    filename = "generated_code.txt"
    if "import " in code_content or "def " in code_content:
        filename = "script.py"
    elif "<html" in code_content.lower():
        filename = "index.html"
    elif "const " in code_content or "let " in code_content:
        filename = "script.js"

    return Response(
        content=code_content,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

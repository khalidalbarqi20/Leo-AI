import os
import requests
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()

# إعداد مجلد الواجهات الرسومية
templates = Jinja2Templates(directory="templates")

# استدعاء مفتاح الـ API المشفر من إعدادات سيرفر Render
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# اسم النموذج المحدث لـ Claude 4.6 Sonnet عبر سيرفر OpenRouter
MODEL_NAME = "anthropic/claude-4.6-sonnet"

@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    # عرض الصفحة فارغة عند أول دخول للموقع
    return templates.TemplateResponse("index.html", {"request": request, "response": ""})

@app.post("/", response_class=HTMLResponse)
async def generate_code(request: Request, prompt: str = Form(...)):
    # التحقق من وجود مفتاح الـ API قبل إرسال الطلب
    if not OPENROUTER_API_KEY:
        error_msg = "خطأ أمني: لم يتم العثور على مفتاح الـ API البرمجي في خادم الاستضافة."
        return templates.TemplateResponse("index.html", {"request": request, "response": error_msg})

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    # الأوامر الثابتة لتهيئة Claude 4.6 ليعمل كخبير برمجي صارم مبني لك خصيصاً
    system_instruction = (
        "أنت مساعد برمجيات ذكي وخبير جداً مخصص للمستخدم فقط. "
        "مهمتك الأساسية هي تلقي المتطلبات باللغة العربية، وكتابة كود برمي كامل، نظيف، "
        "وخالٍ من الأخطاء بناءً على أفضل الممارسات البرمجية لعام 2026. "
        "قم بتنسيق الأكواد داخل بلوكات برمجية واضحة (Markdown Code Blocks) مع توفير شرح مبسط ومباشر باللغة العربية."
    )

    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        # إرسال الطلب عبر السحابة إلى سحابة المطورين للنموذج
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        response_json = response.json()
        
        # استخراج النص البرمجي الراجع من كلاود
        ai_response = response_json['choices'][0]['message']['content']
    except Exception as e:
        ai_response = f"حدث خطأ في الاتصال بالنموذج السحابي: {str(e)}"

    return templates.TemplateResponse("index.html", {"request": request, "response": ai_response, "user_prompt": prompt})

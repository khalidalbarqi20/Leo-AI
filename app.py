# Leo-AI-main/app.py
from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
import os
import logging
import re
import zipfile
import tarfile
import io
import json
import urllib.parse
import aiofiles
import asyncio
import time

# Ø¥Ø¹Ø¯Ø§Ø¯Ø§Øª Ø§ÙØ³Ø¬Ù
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ø¥Ø¹Ø¯Ø§Ø¯Ø§Øª Ø§ÙØªØ·Ø¨ÙÙ
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Ø¥Ø¹Ø¯Ø§Ø¯Ø§Øª CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ø¥Ø¹Ø¯Ø§Ø¯Ø§Øª ÙØ§Ø¦ÙØ© Ø§ÙØ§ÙØªØ¸Ø§Ø±
file_queue = []
executor = ThreadPoolExecutor(max_workers=4)

# Ø¥Ø¹Ø¯Ø§Ø¯Ø§Øª ÙÙØª Ø§ÙØ¨Ø­Ø«
MAX_SEARCH_TIME = 10  # Ø«ÙØ§ÙÙ

# ... (Ø§ÙÙÙØ¯ Ø§ÙÙÙØ¬ÙØ¯ Ø£Ø¹ÙØ§Ù Ø¯ÙÙ ØªØºÙÙØ±)

@app.post("/chat")
async def chat_endpoint(request: Request, model: str = "qwen/qwen3-coder-480b-a35b-instruct:free", message: str = "", files: list[UploadFile] = File(None)):
    """ÙÙØ·Ø© Ø§ÙÙÙØ§ÙØ© Ø§ÙØ±Ø¦ÙØ³ÙØ© ÙÙØ¯Ø±Ø¯Ø´Ø© ÙØ¹ Ø¯Ø¹Ù Ø§ÙÙÙÙØ§Øª"""
    try:
        start_time = time.time()
        
        # ÙØ¹Ø§ÙØ¬Ø© Ø§ÙÙÙÙØ§Øª
        if files:
            for file in files:
                # Ø­ÙØ¸ Ø§ÙÙÙÙ ÙØ¤ÙØªÙØ§
                filename = f"temp_{file.filename}"
                async with aiofiles.open(filename, "wb") as f:
                    await f.write(await file.read())
                file_queue.append(filename)
            
            # ÙØ¹Ø§ÙØ¬Ø© Ø§ÙÙÙÙ ÙÙ Ø®ÙØ· ÙÙÙØµÙ
            executor.submit(process_files, file_queue)
        
        # ÙØ¹Ø§ÙØ¬Ø© Ø§ÙØ±Ø³Ø§ÙØ©
        response = generate_response(model, message)
        
        # Ø§ÙØªØ­ÙÙ ÙÙ ÙÙØª Ø§ÙØ¨Ø­Ø«
        if time.time() - start_time > MAX_SEARCH_TIME:
            logger.warning("Time exceeded for search")
            return JSONResponse({"response": "Ø§ÙÙÙØª ÙØ¯ Ø§ÙØªÙØª. ÙØ±Ø¬Ù Ø¥Ø¹Ø§Ø¯Ø© Ø§ÙÙØ­Ø§ÙÙØ© ÙØ§Ø­ÙÙØ§."})
        
        return JSONResponse({"response": response})
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

async def process_files(filenames):
    """ÙØ¹Ø§ÙØ¬Ø© Ø§ÙÙÙÙØ§Øª ÙÙ Ø®ÙØ· ÙÙÙØµÙ"""
    try:
        for filename in filenames:
            # ÙØ«Ø§Ù: ÙØ±Ø§Ø¡Ø© Ø§ÙÙÙÙ ÙØªØ­ÙÙÙ ÙØ­ØªÙØ§Ù
            async with aiofiles.open(filename, 'rb') as f:
                contents = await f.read()
            
            # ... (ØªØ­ÙÙÙ Ø§ÙÙÙÙ ÙØ¥Ø¶Ø§ÙØ© Ø§ÙÙØªØ§Ø¦Ø¬ Ø¥ÙÙ chat_sessions)
            
            # Ø¥Ø²Ø§ÙØ© Ø§ÙÙÙÙ Ø§ÙÙØ¤ÙØª
            os.remove(filename)
    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")

# ... (Ø§ÙÙÙØ¯ Ø§ÙÙÙØ¬ÙØ¯ Ø£Ø¹ÙØ§Ù Ø¯ÙÙ ØªØºÙÙØ±)

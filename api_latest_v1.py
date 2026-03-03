
# from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
import json,time
import uvicorn
# from hr_bot_v2_new_add_multiquery_add_rerank import HR_BOT
from hr_bot_v2_new_add_multiquery import HR_BOT
# from hr_bot_v2_new import HR_BOT
# from hr_bot_test import HR_BOT
from pydantic import BaseModel
from typing import Dict, Union
from data_structures import Item
import asyncio,re
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from itertools import groupby
from machine_translation import Translator
from threading import Lock
from chatHistoryData import ChatData


app = FastAPI(title="Tietoevry HR AI Service")
translation_count = 0
lock = Lock()

@app.exception_handler(Exception)
async def exception_handler(request:Item , exc: Exception):
    if isinstance(exc, RequestValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
        )
    elif isinstance(exc, ValueError) and str(exc) == "Empty response":
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": "Invalid response"})
    else:
        return JSONResponse(status_code=500, content={"error": "Internal Server Error"})

@app.post("/v2/chat/processor")
async def chat_processor(request: Item):
    print("Begin to create your question...")

     # 解析前端传递的数据
    # properties = request.messages
    print(request.model)
    print(request.stream)
    print(request.citations)
    
    context =""
    for message in request.messages:
        # print(message.role)
        # print(message.content)
        context = message.content


    # 判断是否包含中文字符
    # if re.search('[\u4e00-\u9fa5]', context):
    #     context += ",用中文"
    # else:
    #     context += ",用英文"

    # 设置超时时间（以秒为单位）
    timeout = 30
    # print("message:" + properties["message"])
    # 将解析出的数据透传给大模型并进行处理
    # 模型处理的代码在这里
    bot = HR_BOT()
    try:
      response = await asyncio.wait_for(bot.hr_bot(context), timeout=timeout)
    except asyncio.TimeoutError:
      response = "Timeout occurred. Please try again later."
    response = "success"
    # print("Response:", response)
    # request.result = response
    # return request

    return JSONResponse(content={"message": response})

@app.post("/v1/chats/{chat_id}")
async def generate_chat_completion(chat_id: str, chat_data: ChatData):
    # Print received chat_id and chat_data
    print(chat_id)
    print(chat_data)

    # Access individual fields
    chat = chat_data.chat
    history = chat_data.history

    # Process chat and history data as needed
    # ...

    return {"message": "Chat completion generated"}



@app.post("/v1/chat/completions")
async def completions(request: Item):
    global translation_count
    try:
        print("Begin to create your question...")
        # print(request.model)
        # print(request.stream)
        # print(request.citations)
        
        context =""
        for message in request.messages:
            # print("role:" +message.role)
            # print("content:" +message.content)
            context = message.content

        # if re.search('[\u4e00-\u9fa5]', context):
        #     context += ",用中文"
        # else:
        #     context += ",用英文"

        timeout = 30
        bot = HR_BOT()
        response,sourceList = await asyncio.wait_for(bot.hr_bot(context), timeout=timeout)

        if should_translate(context, response):
        #   translated_response = translate(context, response)
          translated_response = response
          with lock:  # 使用锁确保对全局变量的访问同步
            translation_count += 1
            print(f"translation_count: {translation_count}")  
          # 对翻译后的response进行处理...
        else:
          translated_response = response
          # 对response进行处理...

        # 在这里放置你的处理逻辑
        if translated_response is None:
            raise ValueError("Empty response")
        
        headers = {
            "Content-Type": "text/event-stream; charset=utf-8",
            "Connection": "keep-alive",
        }
        return StreamingResponse(generate_data(translated_response,convert_to_citation(sourceList)), headers=headers)

    except asyncio.TimeoutError:
        return JSONResponse(status_code=408, content={"error": "Timeout occurred. Please try again later."})
    except Exception as e:
        print(f"Error when processing request: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error when processing request")

    
   

def generate_data(content,citations):
    data = {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "gpt-3.5-turbo-0125",
        "system_fingerprint": "fp_44709d6fcb",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": content,
                    
                },
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "citations": citations					
    }
    sse_data = f"data: {json.dumps(data,ensure_ascii=False)}\n\n"
    yield sse_data

@app.get("/v1/models")
def model_list():
    return {
        "object": "list",
        "data": [
            {
                "id": "TietoEvry HR assistant",
                "object": "model",
                "created": 1686935002,
                "owned_by": "organization-owner",
            },
            {
                "id": "model-id-1",
                "object": "model",
                "created": 1686935002,
                "owned_by": "organization-owner",
            },
            {
                "id": "model-id-2",
                "object": "model",
                "created": 1686935002,
                "owned_by": "openai",
            },
        ],
        "object": "list",
    }



@app.get("/v2/chat/processorget")
async def chat_processor():
    print("Begin to read your question...")
    bot = HR_BOT()
    res = bot.hr_bot({"message": "How to add/change phone contact information?"})
    return res

@app.put("/users/{user_id}")
async def update_user(user_id: int, user: dict):
    # 在这里更新用户的逻辑
    return {"message": f"User {user_id} updated successfully"}

@app.delete("/users/{user_id}")
async def delete_user(user_id: int):
    # 在这里删除用户的逻辑
    return {"message": f"User {user_id} deleted successfully"}


def convert_to_citation(data):
    citation = []
    
    for entry in data:
        document_exists = False
        for cit in citation:
            if cit["source"]["name"] == entry["doc_title"]:
                cit["document"].append(entry["page_content"])
                document_exists = True
                break
        
        if not document_exists:
            citation.append({
                "source": {"name": entry["doc_title"]},
                "document": entry["page_content"]
            })
    
    return citation

def convert_to_citation(data):
    citation = []
    
    # 先根据源对数据进行排序
    sorted_data = sorted(data, key=lambda x: x["doc_title"])
    
    
    # 根据源分组数据
    grouped_data = groupby(sorted_data, key=lambda x: x["doc_title"])
   
    
    for source, entries in grouped_data:
        document_list = []
        for entry in entries:
            if source == "N/A":
                document_list.append("")
            else:
                document_list.append(entry["page_content"])
        
        
        citation.append({
            "source": {"name": source},
            "document": document_list
        })
    
    return citation



def should_translate(context, response):
    translator = Translator()
    if translator.detect_language(context) == 'Chinese' and translator.detect_language(response) == 'Chinese':
        # 如果context和response都是中文文本
        return False
    if translator.detect_language(context) == 'English' and translator.detect_language(response) == 'English':
        # 如果context和response都是英文文本
        return False
    return True

def translate(context, response):
    translator = Translator()
    if translator.detect_language(context) == 'Chinese':
        # 如果context是中文，将response翻译成中文
        translated_response = translator.translate_text_one(response, tmp_from_lang="en", tmp_to_lang="zh")
        return translated_response
    else:
        # 如果context是英文，将response翻译成英文
        translated_response = translator.translate_text_one(response, tmp_from_lang="zh", tmp_to_lang="en")
        return translated_response

# Start test server
if __name__ == "__main__":
    uvicorn.run(app, host="10.80.11.197", port=9000)

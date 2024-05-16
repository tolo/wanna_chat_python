from fastapi import FastAPI, Request, Response, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
#import markdown # Maybe
import uuid

from chat_service import ChatService

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

### Using Jinja2 for templating (https://jinja.palletsprojects.com/en/3.0.x/templates/)
templates = Jinja2Templates(directory="templates")

chatService = ChatService()

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    response = templates.TemplateResponse("login.html", {"request": request})
    #response.set_cookie(key="session_key", value=session_key, expires=259200)  # 3 days
    return response


@app.get("/chat", response_class=HTMLResponse)
async def chat(request: Request, user: str):
    messages = chatService.get_history(user)
    context = {
        "request": request,
        "count": len(messages),
        "messages": messages,
    }
    return templates.TemplateResponse("chat.html", context)

@app.post("/chat", response_class=HTMLResponse)
async def add_chat_message(request: Request, text: str = Form(...), user: str = Form(...)):
    answer = chatService.ask_question(user, text)
    messages = chatService.get_history(user)
    context = {
        "request": request,
        "count": len(messages),
        "messages": [{
            "type": "human",
            "text": text,
        }, {
            "type": "ai",
            "text": answer,
        }],
    }
    return templates.TemplateResponse("chat-messages.html", context)






### TODO: API Endpoints


@app.get("/chat")
async def chat(name: str):
    return {"message": f"Hello {name}"}


@app.get("/api/{sessionId}")
async def get_chat_history(name: str):
    return {"message": f"Hello {name}"}

@app.post("/api/{sessionId}")
async def post_chat_message(name: str):
    return {"message": f"Hello {name}"}

@app.delete("/api/{sessionId}")
async def delete_chat_history(name: str):
    return {"message": f"Hello {name}"}

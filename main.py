from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from utils import NarrativeModel

app = FastAPI()
templates = Jinja2Templates(directory="templates")


app = FastAPI()


@app.get("/hello")
def hello():
    return "Hello"


@app.post("/narrative_scanner")
def narrative_scanner(posts: str = Form(), question: str = Form()):
    inputs = posts.split("\n\n")
    model = NarrativeModel(inputs)
    reply = model.generate(question)
    return reply


@app.get("/")
@app.get("/form", response_class=HTMLResponse)
def main(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

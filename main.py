from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from utils import NarrativeModel

app = FastAPI()
templates = Jinja2Templates(directory="templates")


app = FastAPI()


@app.post("/narrative_scanner")
def narrative_scanner(posts: str = Form(), question: str = Form()):
    inputs = posts.split("\n\n")
    model = NarrativeModel(inputs)
    reply = model.generate(question)
    return reply


# @app.get("/")
@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
def root(request: Request):
    # TODO: keep model state between requests. Add IDs to pass to form and select model by it?
    return templates.TemplateResponse("form.html", {"request": request})


@app.post("/form", response_class=HTMLResponse)
def main(request: Request, posts: str = Form(), question: str = Form()):
    inputs = posts.split("\n\n")
    model = NarrativeModel(inputs)
    reply = model.generate(question)
    return templates.TemplateResponse("form.html", {"request": request, "reply": reply})

from fastapi import FastAPI

from utils import NarrativeModel

app = FastAPI()


@app.get("/hello")
def hello():
    return "Hello"


@app.post("/narrative_scanner")
def scan(posts, question):
    inputs = posts.split("\n\n")
    model = NarrativeModel(inputs)
    reply = model.generate(question)
    return reply

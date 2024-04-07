import os

from openai import OpenAI

API_TOKEN = os.environ.get("OPENAPI_TOKEN")
ORG_ID = ""

PROMPT = """
You will be asked questions about a {site} user {user}, who has created the following posts:

###
{inputs}
###

Read the posts carefully and respond truthfully and considering all information provided.
"""


class NarrativeModel:
    def __init__(self, input_texts, model="gpt-3.5-turbo"):
        self.messages = [{"role": "system", "content": self.create_prompt(input_texts)}]
        self.client = OpenAI(
            api_key=API_TOKEN,
        )
        self.model = model

    def generate(self, prompt, generation_args={}):
        self.messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            messages=self.messages, model=self.model, **generation_args  # type: ignore
        )
        self.messages.append(
            {"role": "assistant", "content": response.choices[0].message.content}
        )
        return response.choices[0].message.content

    def create_prompt(self, inputs, site="", user=""):
        inputs = "###\n".join(
            [f"Post {num}:\n{post.strip()}\n" for num, post in enumerate(inputs)]
        )
        return PROMPT.format(inputs=inputs, site=site, user=user)

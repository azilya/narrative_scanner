from langchain.chains import LLMChain
from langchain_community.chat_message_histories import MongoDBChatMessageHistory  # TODO
from langchain.memory import ConversationTokenBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI


SYSTEM_PROMPT = """You will be asked questions about a user, who has created the following posts:
###
{inputs}
###

Read the posts carefully and respond truthfully and considering all information provided."""


class NarrativeModel:
    def __init__(self, input_texts, model="gpt-3.5-turbo"):
        # TODO: add userIDs
        # TODO: create or import by userID
        self._create_new_bubble(input_texts, model)
        self.llm_chain = LLMChain(
            llm=self.llm,
            prompt=self._prompt,
            verbose=True,
            memory=self._memory,
        )

    def _create_new_bubble(self, input_texts, model):
        self._system_prompt = self._create_system_prompt(input_texts)
        self._prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=self._system_prompt
                ),  # The persistent system prompt
                MessagesPlaceholder(
                    variable_name="chat_history"
                ),  # Where the memory will be stored.
                HumanMessagePromptTemplate.from_template(
                    "{human_input}"
                ),  # Where the human input will injected
            ]
        )
        self.llm = ChatOpenAI(model=model)
        self._memory = ConversationTokenBufferMemory(
            llm=self.llm,
            memory_key="chat_history",
            return_messages=True,
        )

    def generate(self, human_input, generation_args={}):
        response = self.llm_chain.predict(human_input)
        return response

    def _create_system_prompt(self, inputs):
        inputs = "###\n".join(
            [f"Post {num}:\n{post.strip()}\n" for num, post in enumerate(inputs)]
        )
        return SYSTEM_PROMPT.format(inputs=inputs)

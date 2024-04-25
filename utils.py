import os
from uuid import uuid4

from langchain.chains import LLMChain
from langchain.memory import ConversationTokenBufferMemory
from langchain.memory.chat_memory import BaseChatMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage, BaseMessage
from langchain_mongodb import MongoDBChatMessageHistory
from langchain_openai import ChatOpenAI

MONGO_URL = os.environ.get("MONGO_URL", "")

# We use system prompt to store info, because it is persistent
# and will not be deleted during history pruning.
# TODO: consider RAG pipeline with text vectors instead?
SYSTEM_PROMPT = """You will be asked questions about a user, \
who has created a series of posts. Carefully read the posts given below, \
between triple dashes, and respond truthfully and considering all information provided.

The posts in question:
```
{inputs}
```
"""


class NarrativeModel:
    def __init__(
        self,
        input_texts: str | None = None,
        session_id: str | None = None,
        model="gpt-3.5-turbo",
    ):
        if not session_id:
            session_id = str(uuid4())
        self.message_store = MongoDBChatMessageHistory(
            connection_string=MONGO_URL,
            session_id=session_id,
            collection_name="message_store",
        )
        self.system_prompts = MongoDBChatMessageHistory(
            connection_string=MONGO_URL,
            session_id=session_id,
            collection_name="system_prompts",
        )
        self.llm = ChatOpenAI(model=model)
        if self.system_prompts.messages == []:
            if not input_texts:
                return "Please enter inputs for bubble generation"
            prompt, memory = self._create_new_bubble(input_texts)
        else:
            prompt, memory = self._load_bubble()
        self.llm_chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            verbose=True,
            memory=memory,
        )

    def _create_prompt_memory(
        self, prompt_message: BaseMessage, prompt_content: str
    ) -> tuple[ChatPromptTemplate, BaseChatMemory]:
        prompt = ChatPromptTemplate.from_messages(
            [
                prompt_message,  # The persistent system prompt
                MessagesPlaceholder(
                    variable_name="chat_history"
                ),  # Where the memory will be stored.
                HumanMessagePromptTemplate.from_template(
                    "{human_input}"
                ),  # Where the human input will injected
            ]
        )
        memory = ConversationTokenBufferMemory(
            llm=self.llm,
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=2000 - self.llm.get_num_tokens(prompt_content),
        )

        return prompt, memory

    def _create_new_bubble(
        self, input_texts: str
    ) -> tuple[ChatPromptTemplate, BaseChatMemory]:
        _system_prompt = self._create_system_prompt(input_texts)
        system_prompt = SystemMessage(content=_system_prompt)
        self.system_prompts.add_message(system_prompt)
        prompt, memory = self._create_prompt_memory(system_prompt, _system_prompt)
        return prompt, memory

    def _load_bubble(self) -> tuple[ChatPromptTemplate, BaseChatMemory]:
        system_prompt = self.system_prompts.messages
        chat_history = self.message_store.messages
        prompt, memory = self._create_prompt_memory(
            system_prompt[0], system_prompt[0].content  # type: ignore
        )
        memory.chat_memory.add_messages(chat_history)
        return prompt, memory

    def generate(self, human_input: str) -> str:
        response = self.llm_chain.predict(human_input=human_input)
        self.message_store.clear()
        self.message_store.add_messages(self.llm_chain.memory.chat_memory.messages)
        # To consider: maybe keep all long history and just use N latest,
        # but for that we need to filter messages that we've already dumped.
        return response

    def _create_system_prompt(self, inputs: str) -> str:
        inputs = "###\n".join(
            [f"Post {num}:\n{post.strip()}\n" for num, post in enumerate(inputs)]
        )
        # TODO: trim prompt to a sensible length, in case input is too long.
        # TODO: ~1024 tokens?
        return SYSTEM_PROMPT.format(inputs=inputs)

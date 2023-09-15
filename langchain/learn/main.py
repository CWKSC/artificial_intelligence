from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.chat_models import ChatOpenAI

repo_id = "databricks/dolly-v2-3b"
llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 64}
)
output = llm("Translate this sentence from English to French: I love programming.")
print(output)

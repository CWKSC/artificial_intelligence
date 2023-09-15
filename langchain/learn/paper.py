import os
os.environ["OPENAI_API_KEY"] = "no key" 

import paperscraper
from paperqa import Docs
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.embeddings import LlamaCppEmbeddings, HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import HuggingFaceHub
from langchain.llms import CTransformers

model = "TheBloke/Llama-2-7b-Chat-GGUF"
llm = CTransformers(model=model)
embeddings = HuggingFaceEmbeddings()
# print(llm('AI is going to'))

docs = Docs(llm=llm, embeddings=embeddings)

keyword_search = 'bispecific antibody manufacture'
papers = paperscraper.search_papers(keyword_search, limit=2)
for path,data in papers.items():
    try:
        docs.add(path, chunk_chars=500)
    except ValueError as e:
        print('Could not read', path, e)

answer = docs.query("What manufacturing challenges are unique to bispecific antibodies?")
print(answer)



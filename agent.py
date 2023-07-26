import os
from typing import Any

import streamlit as st 

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from langchain.agents import Tool, initialize_agent
from langchain.vectorstores import Pinecone
import pinecone
import random

chat = ChatOpenAI(verbose=True, temperature=0)

def calc(val):
    return random.randint(0, 10)

math_tool = Tool(
    name="random",
    func=calc,
    description="If you need a random number then call this method",
)

def find(val):
    return "RM means RobberMan"

rs_tool = Tool(
    name="find about RobberMan",
    func=find,
    description="If you need any information on RobberMan then call this method",
)

template = '''You are a kubernetes wikipedia. Given the user's query {query} on kubernetes comman, suggest some responses. If you don't know the answer, just say that you don't know, don't try to make up an answer.
 
Kubernetes response:'''

prompt = PromptTemplate(
    template=template,
    input_variables=["query"],
)

llm_chain = LLMChain(llm=chat, prompt=prompt, verbose=True)

llm_tool = Tool(
    name="llm",
    func=llm_chain.run,
    description="Use this to generate kubernetes specific queries and logic, return full response",
)

tools = [math_tool, rs_tool, llm_tool]

zero_shot_agent = initialize_agent(
    agent="zero-shot-react-description",
    tools=tools,
    llm=chat,
    verbose=True,
    max_iterations=10,
)

question = st.text_input("Ask something from the file", placeholder="Find something similar to: ....this.... in the text?")    
if question:
    response = zero_shot_agent(question)
    st.write(response)

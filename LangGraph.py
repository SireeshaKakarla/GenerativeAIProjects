from tkinter import Image
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, AnyMessage
from langchain_openai import ChatOpenAI
from ToolCreation import tools
from langgraph.prebuilt import tools_condition, ToolNode
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

openai_api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

llama3 = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama3-8b-8192")    

llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model="gpt-4")

llm_with_tools = llm.bind_tools(tools, tool_choice="auto")

sys_msg = SystemMessage(content="If you are not able to answer the question, do not call any tools. Just say 'I don't know'")

def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
builder.add_edge("assistant", END)

graph = builder.compile()
messages = graph.invoke({"messages": [HumanMessage(content="What's your age")]})
for m in messages['messages']:
    m.pretty_print()


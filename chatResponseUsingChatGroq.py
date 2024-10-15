from dotenv import load_dotenv, find_dotenv
import os
from ToolCreation import tools

load_dotenv(find_dotenv())
groq_api_key = os.getenv("GROQ_API_KEY")


from langchain_groq import ChatGroq

llama3 = ChatGroq(
    groq_api_key=groq_api_key,
    #model_name="llama3-3.1-70b-versatile",
    model_name="llama3-8b-8192",
)

llama3

# Chat with out tools
#print(llama3.invoke("add two numbers 11 and 25"))

# Bind tools with LLM
from langchain_core.tools import tool

llama3_with_tools = llama3.bind_tools(tools, tool_choice="auto")

# Chat with tools
# print(llama3_with_tools.invoke("What is the address of Apple Inc.?"))
print(llama3_with_tools.invoke("Add two numbers 10 and 20"))
import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
load_dotenv()

llm = ChatMistralAI(model="mistral-large-latest")


while True:
    if query == "exit":
        break
    query = input("User: ")
response = llm.invoke(query)
print(f"Bot AI: {response.content}")
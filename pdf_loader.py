import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(
    model="deepseek/deepseek-chat",  
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7
)


prompt = "Hello! Please reply with exactly one short sentence."

print("Sending a tiny test message to OpenRouter...")

response = model.invoke(prompt)

print("\n--- OPENROUTER RESPONSE ---")
print(response.content)
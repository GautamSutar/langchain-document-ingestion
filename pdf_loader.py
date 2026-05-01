import os
from google import genai
from dotenv import load_dotenv
import time 
from langchain_community.document_loaders import PyPDFLoader
load_dotenv()

# client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

# print("Extracting PDF")
# loader = PyPDFLoader("Git.pdf")
# docs = loader.load()

# page_1_text = docs[0].page_content
# page_2_text = docs[1].page_content 
# total_pages = len(docs)

# prompt = f"""
# I am analyzing a document that is {total_pages} pages long. 

# Here is the exact text from Page 1:
# {page_1_text}

# Here is the exact text from Page 2:
# {page_2_text}

# Based on this text, provide the following:
# 1. The total number of pages in this document.
# 2. The exact text content from the first page.
# 3. A summary of the metadata or main topics found specifically on the second page.
# """

# print("Sending extracted text to Gemini...")

# print("Fetching Allowed Models")
# for model in client.models.list():
#     if "flash" in model.name:
#         print(model.name)
# # 5. Send the much smaller text payload to Gemini 1.5
# response = client.models.generate_content(
#     model='gemini-2.0-flash-lite',
#     contents=prompt
# )

# print("\n--- GEMINI RESPONSE ---")
# print(response.text)


client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

# 2. Create a tiny, low-token prompt (no PDF!)
prompt = "Hello! Please reply with exactly one short sentence."

print("Sending a tiny test message to Gemini...")

# 3. Send to the Lite model
response = client.models.generate_content(
    model='gemini-2.0-flash-lite',
    contents=prompt
)

print("\n--- GEMINI RESPONSE ---")
print(response.text)
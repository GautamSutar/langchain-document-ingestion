import os
from google import genai
from dotenv import load_dotenv
import time 
from langchain_community.document_loaders import PyPDFLoader
load_dotenv()

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

pdf_document = client.files.upload(file="Git.pdf")

prompt = """
Analyze the attached PDF and provide the following:
1. The total number of pages in this document.
2. The exact text content from the first page.
3. A summary of the metadata or main topics found specifically on the second page.
"""
time.sleep(4)
response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents=[pdf_document, prompt]
)

print(response.text)

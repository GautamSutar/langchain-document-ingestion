from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.0-flash-lite')

prompt = PromptTemplate(
    template='Write a summary for the following poem - \n {poem}',
    input_variables=['poem']
)

parser = StrOutputParser()

loader = TextLoader('football.txt', encoding='utf-8')
docs = loader.load()

print(f"Type     : {type(docs)}")
print(f"Num Docs : {len(docs)}")
print(f"Metadata : {docs[0].metadata}")
print(f"Content  :\n{docs[0].page_content[:500]}...")

chain = prompt | model | parser

print(chain.invoke({'poem':docs[0].page_content}))
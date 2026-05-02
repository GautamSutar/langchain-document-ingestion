from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI(
    model="deepseek/deepseek-chat",
    temperature=0.7,
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),

)

prompt = PromptTemplate(
    template='Write a summary for the following poem - \n {poem}',
    input_variables=['poem']
)

parser = StrOutputParser()

loader = TextLoader('football.txt', encoding='utf-8')
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(docs)

chain = prompt | model | parser

print("\nSummary of the first chunk:")
print(chain.invoke({'poem': chunks[0].page_content}))
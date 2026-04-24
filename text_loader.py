from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
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

# Document Splitting (The next step in ingestion)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(docs)

print(f"Num Chunks: {len(chunks)}")
print(f"Chunk 1 Content:\n{chunks[0].page_content}")

chain = prompt | model | parser

# Invoke on the first chunk since the model might have quota issues with large text
print("\nSummary of the first chunk:")
print(chain.invoke({'poem': chunks[0].page_content}))
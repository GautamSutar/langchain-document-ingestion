from langchain_community.document_loaders import WebBaseLoader
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
    template='Answer the following questions \n {question} from the following text - \n {text}',
    input_variables=['question', 'text']
)

parser = StrOutputParser()
url = "https://www.geeksforgeeks.org/courses"
loader = WebBaseLoader(url) 
docs = loader.load()
chain = prompt | model | parser 
print(chain.invoke({'question': 'How many courses are there?', 'text': docs[0].page_content}))

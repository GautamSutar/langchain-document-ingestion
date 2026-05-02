from langchain_community.document_loaders import CSVLoader 
from langchain_openai import ChatOpenAI 
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser 
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
loader = CSVLoader(file_path='cancer_patients.csv')
docs = loader.load() 

chain = prompt | model | parser
print(chain.invoke({'question': 'print all patients name and cancer type with details', 'text': docs}))
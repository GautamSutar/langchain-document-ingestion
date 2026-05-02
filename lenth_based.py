from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Git.pdf")
docs = loader.load() 


# text= """Technology has become an integral part of modern life, transforming the way we communicate, work, and solve problems. From artificial intelligence and cloud computing to smartphones and the Internet of Things, technological advancements are continuously improving efficiency and connectivity across the globe. Businesses leverage data analytics and automation to make smarter decisions, while individuals benefit from instant access to information and digital services. At the same time, rapid innovation also brings challenges such as data privacy, cybersecurity, and ethical concerns around AI. As technology continues to evolve, its responsible use will play a crucial role in shaping a more innovative, inclusive, and sustainable future.
# """

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=50,
    separator=''
    )    
result = splitter.split_documents(docs)
print(result[2].page_content)
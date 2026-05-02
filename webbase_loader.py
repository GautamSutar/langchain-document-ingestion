from langchain_community.document_loaders import WebBaseLoader

url = "https://www.mptourism.com/destination-bhimbetka.php"

loader = WebBaseLoader(url) 
docs = loader.load()
print(docs) 
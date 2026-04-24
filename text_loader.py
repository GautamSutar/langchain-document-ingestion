from langchain_community.document_loaders import TextLoader

loader = TextLoader('football.txt', encoding='utf-8')
docs = loader.load()

print(f"Type     : {type(docs)}")
print(f"Num Docs : {len(docs)}")
print(f"Metadata : {docs[0].metadata}")
print(f"Content  :\n{docs[0].page_content[:500]}...")
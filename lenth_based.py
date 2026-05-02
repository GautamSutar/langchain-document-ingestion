from langchain_text_splitters import CharacterTextSplitter

text= """Technology has become an integral part of modern life, transforming the way we communicate, work, and solve problems. From artificial intelligence and cloud computing to smartphones and the Internet of Things, technological advancements are continuously improving efficiency and connectivity across the globe. Businesses leverage data analytics and automation to make smarter decisions, while individuals benefit from instant access to information and digital services. At the same time, rapid innovation also brings challenges such as data privacy, cybersecurity, and ethical concerns around AI. As technology continues to evolve, its responsible use will play a crucial role in shaping a more innovative, inclusive, and sustainable future.
"""

splitter = CharacterTextSplitter(
    chunk_size=20,
    chunk_overlap=0,
    separator=''
    )    
result = splitter.split_text(text)
print(result)
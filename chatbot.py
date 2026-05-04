from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("/Users/yuguo/Downloads/House-owner-wordings.pdf")
pages = loader.load()


print(pages[0].page_content)
print(pages[0].metadata)


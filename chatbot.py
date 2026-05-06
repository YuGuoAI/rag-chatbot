from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()


# Api set up and test
client = OpenAI(api_key=os.getenv(key='OPENAI_API_KEY'))
response = client.responses.create(
    model='gpt-5',
    input='tell me how human brain works(within 20 words)'
)

# load the documents
loader = PyPDFLoader("/Users/yuguo/Downloads/House-owner-wordings.pdf")
pages = loader.load()

# chunk the content
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = splitter.split_documents(pages)

# Verify
print(f"Pages loaded: {len(pages)}")
print(f"Chunks created: {len(chunks)}")

encoding = tiktoken.encoding_for_model('text-embedding-3-small')
tokens = encoding.encode(" the")


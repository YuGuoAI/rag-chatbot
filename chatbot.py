from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
import langchain_pinecone


load_dotenv()

# Api keys setup
client = OpenAI(api_key=os.getenv(key='OPENAI_API_KEY'))
pc = Pinecone(api_key=os.getenv(key='PINECONE_API_KEY'))

# load the documents
loader = PyPDFLoader("/Users/yuguo/Downloads/House-owner-wordings.pdf")
pages = loader.load()

# chunk the content
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = splitter.split_documents(pages)

# verify
print(f"Pages loaded: {len(pages)}")
print(f"Chunks created: {len(chunks)}")

# langchain openai embedding
'''response = client.embeddings.create(
    input=chunks[0].page_content,
    model="text-embedding-3-small"
)
print(f"Number of chunks: {len(chunks)}")'''

embedding = OpenAIEmbeddings(
    model='text-embedding-3-small',
)
chunks_text = [chunk.page_content for chunk in chunks]
all_vectors = embedding.embed_documents(texts=chunks_text)

# pinecone index created
index_name = "rag-chatbot"

if index_name not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536, 
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        deletion_protection='enabled',
    )
    print(f"✅ Created index '{index_name}'")
else:
    print(f"ℹ️  Index '{index_name}' already exists")


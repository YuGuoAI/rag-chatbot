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
chunks_metadata = [chunk.metadata for chunk in chunks]
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

# load in batch
## helper function
def batch_iterator(items, batch_size=100):
    """Help to batch the chunks"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

def build_metadata(chunk):
    """To make the page in the metadata starts form 1 instead of 0.
    And return all the metadata we want to upsert."""
    page = chunk.metadata.get("page")
    if page is not None:
        chunk.metadata["page"] = page + 1
    metadata = {
        "text": chunk.page_content,
        "source": chunk.metadata.get("source", "unknown"),
        "page" : chunk.metadata.get("page", "unknown"),
        "total_pages" : chunk.metadata.get("total_pages", "unknown")
    }
    return metadata

test = chunks[1].metadata.get("source","")

## Build records (metadata + text)
records = [
    {
        "id": f"chunk-{i}",
        "values": vector,
        "metadata": build_metadata(chunk),
    }
    for i, (chunk, vector) in enumerate(zip(chunks, all_vectors))
]

## load
index = pc.Index(host=os.getenv("INDEX_HOST"))

for i, batch in enumerate(batch_iterator(records, batch_size=100), start=1):
    index.upsert(vectors=batch, namespace='Insurance')
    print(f"✅ Batch {i} done — {min(i * 100, len(records))}/{len(records)} vectors")

# question test


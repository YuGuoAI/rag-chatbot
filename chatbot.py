from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

# load the env
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
embedding = OpenAIEmbeddings(
    model='text-embedding-3-small',
)
chunks_text = [chunk.page_content for chunk in chunks]
chunks_metadata = [chunk.metadata for chunk in chunks]
all_vectors = embedding.embed_documents(texts=chunks_text)

# pinecone index created
index_name = os.getenv("INDEX_NAME")

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
question = "What is the deductible amount?"

search_vector = embedding.embed_query(
    text=question,
)

search_result = index.query(
    namespace="Insurance",
    vector=search_vector, 
    top_k=3,
    include_metadata=True,
    include_values=False
)

print(f"\nQuestion: {question}\n")
print(f"Found {len(search_result['matches'])} matches:\n")

for i, match in enumerate(search_result['matches'], start=1):
    print(f"--- Match {i} (similarity score: {match['score']:.3f}) ---")
    print(f"Source: {match['metadata'].get('source', 'unknown')}")
    print(f"Page: {match['metadata'].get('page', 'unknown')}")
    print(f"Text preview: {match['metadata']['text'][:200]}...")
    print()

# extract the text from each match
context_chunks = [match['metadata']['text'] for match in search_result['matches']]
context = "\n\n---\n\n".join(context_chunks)
# Build numbered context for citations

SCORE_THRESHOLD = 0.4
relevant_matches = [
    m for m in search_result['matches'] 
    if m['score'] >= SCORE_THRESHOLD
]

numbered_context = "\n\n".join([
    f"[{i+1}] (Source: {m['metadata'].get('source', 'unknown').split('/')[-1]}, "
    f"Page: {m['metadata'].get('page', 'unknown')})\n"
    f"{m['metadata']['text']}"
    for i, m in enumerate(relevant_matches)
])

prompt = f"""You are an assistant that answers questions based ONLY on provided context.

STRICT RULES:
1. Use ONLY information explicitly stated in the numbered context chunks below.
2. For every claim, cite the chunk number in brackets like [1] or [2].
3. If you cannot find the answer in the context, respond exactly: "I cannot answer this question based on the provided documents."
4. Do NOT use prior knowledge or make assumptions beyond what the context says.
5. If sources conflict, point out the conflict instead of choosing one.

Context:
{numbered_context}

Question: {question}

Answer (with citations):"""



if not relevant_matches:
    print(f"\n💬 Question: {question}")
    print(f"\n📝 Answer: I don't have relevant information in my knowledge base to answer this question confidently.")
else:
    response = client.responses.create(
        model= "gpt-5",
        input= prompt,
    )

answer = response.output_text

# Display the answer with citations
print(f"\n💬 Question: {question}")
print(f"\n📝 Answer: {answer}")
print(f"\n📚 Sources used:")
for match in search_result['matches']:
    source = match['metadata'].get('source', 'unknown').split('/')[-1]
    page = match['metadata'].get('page', 'unknown')
    print(f"   • {source}, page {page} (score: {match['score']:.3f})")
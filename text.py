# Connect to the index
index = pc.Index(index_name)

# Prepare records: id + vector + metadata
records = []
for i, (chunk, vector) in enumerate(zip(chunks, all_vectors)):
    records.append({
        "id": f"chunk-{i}",
        "values": vector,
        "metadata": {
            "text": chunk.page_content,
            "source": chunk.metadata.get("source", ""),
            "page": chunk.metadata.get("page", 0),
        }
    })

# Upload in batches of 100
batch_size = 100
for i in range(0, len(records), batch_size):
    batch = records[i:i + batch_size]
    index.upsert(vectors=batch)
    print(f"Uploaded {i + len(batch)} / {len(records)} records")

# Verify
stats = index.describe_index_stats()
print(f"\n✅ Index now contains {stats['total_vector_count']} vectors")




def batch_iterator(items, batch_size=100):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

# Build records (metadata + text)
records = [
    {
        "id": f"chunk-{i}",
        "values": vector,
        "metadata": {"text": chunk.page_content, "source": chunk.metadata.get("source", "")},
    }
    for i, (chunk, vector) in enumerate(zip(chunks, all_vectors))
]

# Upload in batches
for batch in batch_iterator(records, batch_size=100):
    index.upsert(vectors=batch)
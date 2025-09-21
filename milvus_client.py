# milvus_client.py
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from embedding import get_embedding
import os
from dotenv import load_dotenv

load_dotenv()

# =====================
# Cấu hình
# =====================
COLLECTION_NAME = "course_chunks"
DIM = 384  # intfloat/multilingual-e5-small có dim=384

# Kết nối Milvus
connections.connect("default", host=os.getenv("DB_HOST"), port="19530")

# =====================
# Tạo Collection
# =====================
def create_collection():
    if utility.has_collection(COLLECTION_NAME):
        collection = Collection(COLLECTION_NAME)
        collection.load()
        return collection

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="course_id", dtype=DataType.INT64),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIM),
    ]

    schema = CollectionSchema(fields, description="Course Chunks Embeddings")
    collection = Collection(COLLECTION_NAME, schema, consistency_level="Strong")

    # Tạo index cho vector search
    collection.create_index(
        field_name="embedding",
        index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 100}}
    )
    collection.load()
    return collection

# =====================
# Insert dữ liệu
# =====================
def insert_course_chunks(course_id: int, chunks: list[str]):
    collection = Collection(COLLECTION_NAME)

    # Dữ liệu chỉ gửi 4 field, bỏ id auto
    data = [
        [],  # course_id
        [],  # chunk_index
        [],  # text
        [],  # embedding
    ]

    for idx, chunk in enumerate(chunks):
        emb = get_embedding(chunk, is_query=False)
        data[0].append(course_id)     # course_id
        data[1].append(idx)           # chunk_index
        data[2].append(chunk)         # text
        data[3].append(emb)           # embedding

        # Debug info
        if idx == 0:
            print("Embedding dim check:", len(emb))  # phải = 384

    collection.insert(data)
    collection.flush()
    collection.load()
    print(f"Inserted {len(chunks)} chunks. Total entities:", collection.num_entities)

# =====================
# Search
# =====================
def search(query: str, top_k=3):
    collection = Collection(COLLECTION_NAME)
    embedding = get_embedding(query, is_query=True)

    results = collection.search(
        [embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["text", "course_id", "chunk_index"]
    )

    hits = []
    for hit in results[0]:
        hits.append({
            "text": hit.entity.get("text"),
            "course_id": hit.entity.get("course_id"),
            "chunk_index": hit.entity.get("chunk_index"),
            "score": hit.score
        })
    return hits

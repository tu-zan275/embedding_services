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
        FieldSchema(name="course_name", dtype=DataType.VARCHAR, max_length=255),  # metadata
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
def insert_course_chunks(course_id: int, chunks: list[str], course_name: str):
    collection = Collection(COLLECTION_NAME)

    # Dữ liệu chỉ gửi 5 field, bỏ id auto
    data = [
        [],  # course_id
        [],  # chunk_index
        [],  # text
        [],  # embedding
        [],  # course_name
    ]

    for idx, chunk in enumerate(chunks):
        emb = get_embedding(chunk, is_query=False)
        data[0].append(course_id)     # course_id
        data[1].append(idx)           # chunk_index
        data[2].append(chunk)         # text
        data[3].append(emb)           # embedding
        data[4].append(course_name)   # course_name

        # Debug info
        if idx == 0:
            print("Embedding dim check:", len(emb))  # phải = 384

    collection.insert(data)
    collection.flush()
    collection.load()
    print(f"Inserted {len(chunks)} chunks. Total entities:", collection.num_entities)

# =====================
# Search + build context
# =====================
def search(query: str, top_k=5):
    """
    Tìm kiếm các chunks gần nhất với query, trả về context đầy đủ cho AI.
    """
    collection = Collection(COLLECTION_NAME)
    embedding = get_embedding(query, is_query=True)

    # Nếu bạn thêm category metadata, hãy include vào output_fields
    results = collection.search(
        [embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["text", "course_id", "chunk_index", "course_name"]  # + "category" nếu có
    )

    contexts = []
    hits = []

    for hit in results[0]:
        course_name = hit.entity.get("course_name")
        chunk_index = hit.entity.get("chunk_index")
        text = hit.entity.get("text")
        course_id = hit.entity.get("course_id")
        score = hit.score

        # Build context string cho AI
        ctx = f"Course: {course_name} (ID: {course_id})\nChunk {chunk_index}: {text}"
        contexts.append(ctx)

        # Lưu thông tin chi tiết cho debug hoặc trả về API
        hits.append({
            "course_name": course_name,
            "course_id": course_id,
            "chunk_index": chunk_index,
            "text": text,
            "score": score
        })

    # Kết hợp các context lại
    context_text = "\n\n".join(contexts)

    return {
        "context_text": context_text,  # dùng để đưa vào LLM
        "hits": hits                    # chi tiết từng hit, nếu cần
    }

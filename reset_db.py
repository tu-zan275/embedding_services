from pymilvus import utility, FieldSchema, CollectionSchema, DataType, Collection, connections
import os
from dotenv import load_dotenv

load_dotenv()
_DB_HOST = os.getenv("DB_HOST")
connections.connect("default", host=_DB_HOST, port="19530")

# Xóa collection cũ
if utility.has_collection("course_chunks"):
    utility.drop_collection("course_chunks")

# Tạo schema mới
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="course_id", dtype=DataType.INT64),
    FieldSchema(name="chunk_index", dtype=DataType.INT64),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4000),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
    FieldSchema(name="course_name", dtype=DataType.VARCHAR, max_length=500),
]

schema = CollectionSchema(fields, description="Course Chunks Embeddings")
collection = Collection("course_chunks", schema, consistency_level="Strong")
collection.create_index(
    field_name="embedding",
    index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 100}}
)
collection.load()
print("Collection created with 5 fields.")

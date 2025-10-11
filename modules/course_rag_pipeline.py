# course_rag_pipeline.py
import os
from dotenv import load_dotenv
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer
import uuid

load_dotenv()

# 1️⃣ Kết nối Milvus
#connections.connect("default", host="localhost", port="19530")
connections.connect("default", host=os.getenv("MILVUS_HOST"), port=os.getenv("MILVUS_PORT"))

# 2️⃣ Tạo schema unified cho cả khóa học & bài học
def create_course_rag_collection():
    collection_name = "course_rag"
    if utility.has_collection(collection_name):
        print("Collection đã tồn tại, skip tạo.")
        return Collection(collection_name)

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=10),  # 'course' hoặc 'lesson'
        FieldSchema(name="course_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="course_title", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="lesson_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="lesson_title", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="author", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=512),
    ]
    schema = CollectionSchema(fields, description="Unified RAG schema for courses and lessons")
    collection = Collection(name=collection_name, schema=schema)

    # Tạo index vector để search nhanh
    index_params = {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 128}}
    collection.create_index(field_name="embedding", index_params=index_params)
    print("✅ Collection created:", collection_name)
    return collection

#embed_query = SentenceTransformer("intfloat/multilingual-e5-small")  # dùng cho truy vấn
#embed_corpus = SentenceTransformer("intfloat/multilingual-e5-base")  # dùng cho indexing nội dung

# 3️⃣ Sinh embedding model (Việt hóa hoặc đa ngôn ngữ)
embed_model = SentenceTransformer("intfloat/multilingual-e5-base")

def embed_text(text):
    return embed_model.encode(text).tolist()

# 4️⃣ Chuẩn bị dữ liệu mẫu
courses = [
    {
        "course_id": "C001",
        "title": "Khóa học SEO nâng cao",
        "author": "Nguyễn Minh",
        "category": "Marketing",
        "description": "Khóa học giúp bạn tối ưu SEO website, nghiên cứu từ khóa và xây dựng nội dung chất lượng.",
        "url": "https://example.com/seo-nang-cao",
        "lessons": [
            {"lesson_id": "L001", "title": "Nghiên cứu từ khóa", "content": "Cách phân tích và chọn từ khóa phù hợp."},
            {"lesson_id": "L002", "title": "Tối ưu Onpage", "content": "Kỹ thuật tối ưu tiêu đề, meta và heading."},
            {"lesson_id": "L003", "title": "Xây dựng backlink", "content": "Phương pháp tạo liên kết chất lượng cao."}
        ]
    },
    {
        "course_id": "C002",
        "title": "Khóa học Lập trình Python cơ bản",
        "author": "Trần Huy",
        "category": "Lập trình",
        "description": "Học Python từ căn bản đến nâng cao, qua ví dụ thực tế và bài tập ứng dụng.",
        "url": "https://example.com/python-co-ban",
        "lessons": [
            {"lesson_id": "L004", "title": "Giới thiệu Python", "content": "Lịch sử và ứng dụng thực tế của Python."},
            {"lesson_id": "L005", "title": "Biến và kiểu dữ liệu", "content": "Cách khai báo và sử dụng biến trong Python."}
        ]
    }
]


# 5️⃣ Chuyển dữ liệu thành các bản ghi embedding
def prepare_records(courses):
    records = []
    for course in courses:
        # Khóa học
        course_text = f"Khóa học: {course['title']}. Tác giả: {course['author']}. Danh mục: {course['category']}. Nội dung: {course['description']}"
        records.append({
            "id": str(uuid.uuid4()),
            "embedding": embed_text(course_text),
            "type": "course",
            "course_id": course["course_id"],
            "course_title": course["title"],
            "lesson_id": "",
            "lesson_title": "",
            "author": course["author"],
            "category": course["category"],
            "content": course["description"],
            "url": course["url"]
        })

        # Các bài học
        for lesson in course["lessons"]:
            lesson_text = f"Bài học: {lesson['title']}. Thuộc khóa học: {course['title']}. Tác giả: {course['author']}. Nội dung: {lesson['content']}"
            records.append({
                "id": str(uuid.uuid4()),
                "embedding": embed_text(lesson_text),
                "type": "lesson",
                "course_id": course["course_id"],
                "course_title": course["title"],
                "lesson_id": lesson["lesson_id"],
                "lesson_title": lesson["title"],
                "author": course["author"],
                "category": course["category"],
                "content": lesson["content"],
                "url": course["url"]
            })
    return records


# 6️⃣ Insert vào Milvus
def insert_data(collection, records):
    data = [
        [r["id"] for r in records],
        [r["embedding"] for r in records],
        [r["type"] for r in records],
        [r["course_id"] for r in records],
        [r["course_title"] for r in records],
        [r["lesson_id"] for r in records],
        [r["lesson_title"] for r in records],
        [r["author"] for r in records],
        [r["category"] for r in records],
        [r["content"] for r in records],
        [r["url"] for r in records],
    ]
    collection.insert(data)
    collection.flush()
    print(f"✅ Đã insert {len(records)} records vào Milvus")


# 7️⃣ Tìm kiếm RAG (semantic query)
def query_rag(collection, query, filter_expr=None, limit=5):
    q_emb = embed_text(query)
    results = collection.search(
        data=[q_emb],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"nprobe": 8}},
        limit=limit,
        expr=filter_expr,
        output_fields=["type", "course_title", "lesson_title", "author", "url", "content"]
    )
    print(f"\n🔍 Query: {query}\n")
    for hit in results[0]:
        print(f"[{hit.entity.get('type').upper()}] {hit.entity.get('course_title')} → {hit.entity.get('lesson_title')}")
        print(f"Tác giả: {hit.entity.get('author')} | URL: {hit.entity.get('url')}")
        print(f"Nội dung: {hit.entity.get('content')[:100]}...\n")


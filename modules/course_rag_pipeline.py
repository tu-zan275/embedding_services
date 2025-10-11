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
        collection = Collection(collection_name)
    else:
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=10),
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

        index_params = {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 128}}
        collection.create_index(field_name="embedding", index_params=index_params)
        print("✅ Collection created:", collection_name)

    # 🔥 Load collection vào RAM để có thể search
    try:
        collection.load()
        print("✅ Collection loaded vào RAM.")
    except Exception as e:
        print("⚠️ Load collection thất bại:", e)

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


# =====================
# QUERY RAG — optimized
# =====================
def query_rag_v1(collection, query, filter_expr=None, limit=5):
    """
    Tìm kiếm ngữ nghĩa (semantic search) trên Milvus.
    Trả về danh sách dict dễ dùng cho API.
    """
    q_emb = embed_text(query)

    results = collection.search(
        data=[q_emb],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"nprobe": 8}},
        limit=limit,
        expr=filter_expr,
        output_fields=["type", "course_title", "lesson_title", "author", "url", "content"]
    )

    hits = []
    print(f"\n🔍 Query: {query}\n")
    for hit in results[0]:
        item = {
            "score": hit.score,
            "type": hit.entity.get("type"),
            "course_title": hit.entity.get("course_title"),
            "lesson_title": hit.entity.get("lesson_title"),
            "author": hit.entity.get("author"),
            "url": hit.entity.get("url"),
            "content": hit.entity.get("content"),
        }
        hits.append(item)

        # Log ngắn gọn
        print(f"[{item['type'].upper()}] {item['course_title']} → {item['lesson_title']}")
        print(f"Tác giả: {item['author']} | URL: {item['url']}")
        print(f"Nội dung: {item['content'][:100]}...\n")

    print(f"✅ Tổng số kết quả: {len(hits)}\n")
    return hits


def query_rag(collection, query, filter_expr=None, limit=5, input_search_type):
    """
    Truy vấn semantic search trên Milvus.
    - Nếu query mang tính tổng quan -> tìm 'course'
    - Nếu query mang tính chi tiết về bài học -> tìm 'lesson'
    """

    query_lower = query.lower()

    lesson_keywords = ["bài học", "lesson", "chương", "phần", "nội dung", "cách học", "hướng dẫn", "ví dụ"]

    # if any(k in query_lower for k in lesson_keywords):
    #     search_type = "lesson"
    # else:
    #     search_type = "course"

    search_type = input_search_type

    q_emb = embed_text(query)

    expr = f"type == '{search_type}'"
    if filter_expr:
        expr += f" and {filter_expr}"

    results = collection.search(
        data=[q_emb],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"nprobe": 8}},
        limit=limit,
        expr=expr,
        output_fields=[
            "type", "course_title", "lesson_title",
            "author", "category", "url", "content"
        ]
    )

    hits = []
    print(f"\n🔍 Query: {query}  →  Tìm trong: {search_type.upper()}\n")

    for hit in results[0]:
        entity = hit.entity
        item = {
            "score": round(hit.score, 4),
            "type": entity.get("type"),
            "course_title": entity.get("course_title"),
            "lesson_title": entity.get("lesson_title"),
            "author": entity.get("author"),
            "category": entity.get("category"),
            "url": entity.get("url"),
            "content": entity.get("content"),
        }
        hits.append(item)

        label = item["lesson_title"] if item["type"] == "lesson" else item["course_title"]
        print(f"[{item['type'].upper()}] {item['course_title']} → {item['lesson_title'] or 'N/A'}")
        print(f"Tác giả: {item['author']} | URL: {item['url']}")
        print(f"Nội dung: {item['content'][:120]}...\n")

    print(f"✅ Tổng kết quả: {len(hits)}\n")
    return hits


## V3
# import numpy as np

# # =====================================
# # 🧩 Huấn luyện sơ bộ hướng câu hỏi
# # =====================================
# course_examples = [
#     "Khóa học này nói về gì?",
#     "Ai là giảng viên của khóa học?",
#     "Tôi nên học khóa nào về Python?",
#     "Khóa học nào giúp tôi nâng cao kỹ năng lập trình?",
# ]

# lesson_examples = [
#     "Bài học đầu tiên dạy cái gì?",
#     "Trong chương 2 có hướng dẫn thực hành không?",
#     "Nội dung của bài học này là gì?",
#     "Bài 3 nói về cách cài đặt ra sao?",
# ]

# course_query_vector = np.mean([embed_text(q) for q in course_examples], axis=0)
# lesson_query_vector = np.mean([embed_text(q) for q in lesson_examples], axis=0)


# # =====================================
# # 🚀 Hàm query chính
# # =====================================
# def query_rag(collection, query, filter_expr=None, limit=5):
#     """
#     Tìm kiếm semantic RAG trên Milvus, tự nhận diện loại câu hỏi.
#     """
#     q_emb = embed_text(query)

#     # 1️⃣ Phân loại câu hỏi bằng cosine similarity
#     def cosine_sim(a, b):
#         return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

#     sim_course = cosine_sim(q_emb, course_query_vector)
#     sim_lesson = cosine_sim(q_emb, lesson_query_vector)
#     search_type = "lesson" if sim_lesson > sim_course else "course"

#     # 2️⃣ Chuẩn bị filter
#     expr = f"type == '{search_type}'"
#     if filter_expr:
#         expr += f" and {filter_expr}"

#     # 3️⃣ Truy vấn Milvus
#     results = collection.search(
#         data=[q_emb],
#         anns_field="embedding",
#         param={"metric_type": "IP", "params": {"nprobe": 8}},
#         limit=limit,
#         expr=expr,
#         output_fields=[
#             "type", "course_title", "lesson_title",
#             "author", "category", "url", "content"
#         ]
#     )

#     # 4️⃣ Kết quả
#     hits = []
#     print(f"\n🔍 Query: {query}")
#     print(f"🤖 Phân loại: {search_type.upper()} (sim_course={sim_course:.3f}, sim_lesson={sim_lesson:.3f})\n")

#     for hit in results[0]:
#         e = hit.entity
#         hits.append({
#             "score": round(hit.score, 4),
#             "type": e.get("type"),
#             "course_title": e.get("course_title"),
#             "lesson_title": e.get("lesson_title"),
#             "author": e.get("author"),
#             "category": e.get("category"),
#             "url": e.get("url"),
#             "content": e.get("content"),
#         })

#         # Log ngắn
#         print(f"[{e.get('type').upper()}] {e.get('course_title')} → {e.get('lesson_title') or 'N/A'}")
#         print(f"URL: {e.get('url')}")
#         print(f"Nội dung: {e.get('content')[:120]}...\n")

#     print(f"✅ Tổng: {len(hits)} kết quả\n")
#     return hits

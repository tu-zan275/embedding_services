
# main.py
from fastapi import FastAPI
from rag_service import rag_answer_v2
from models.requests.InsertPayload import InsertPayload
from models.requests.AskPayload import AskPayload

from collections.course_rag_pipeline import (
    create_course_rag_collection,
    prepare_records,
    insert_data,
    query_rag,
    courses  # nếu muốn dùng dataset mẫu
)

app = FastAPI()

# 1️⃣ Khởi tạo collection (chỉ chạy 1 lần)
collection = create_course_rag_collection()

# # 2️⃣ Chuẩn bị dữ liệu mẫu (hoặc load từ DB)
# records = prepare_records(courses)

# # 3️⃣ Insert vào Milvus
# insert_data(collection, records)

# 4️⃣ Gọi truy vấn RAG
# query_rag(collection, "Tìm khóa học về SEO nâng cao")
# query_rag(collection, "Bài học tối ưu onpage thuộc khóa nào?")
# query_rag(collection, "Khóa học Python có bao nhiêu bài học?")

# if __name__ == "__main__":
#     collection = create_course_rag_collection()
#     records = prepare_records(courses)
#     insert_data(collection, records)

#     # Thử truy vấn RAG
#     query_rag(collection, "Khóa học SEO nâng cao có những bài học nào?")
#     query_rag(collection, "Bài học tối ưu onpage thuộc khóa nào?")
#     query_rag(collection, "Tìm khóa học về lập trình Python")

@app.get("/")
def root():
    return {"status": "ok", "message": "RAG API is running"}

@app.post("/insert")
def insert(payload: InsertPayload):
    try:
        # 1️⃣ Convert payload sang dict
        course_dict = payload.dict()

        # 2️⃣ Chuẩn bị dữ liệu embedding (pipeline)
        records = prepare_records([course_dict]) 

        # 3️⃣ Insert vào Milvus
        insert_data(collection, records)

        return {
            "status": "ok",
            "course_id": payload.course_id,
            "chunks_count": len(records)
        }

    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.post("/ask")
def ask_question(payload: AskPayload):
    try:
        result = rag_answer_v2(payload.query, top_k=payload.top_k)
        return {
            "status": "ok",
            "query": result["query"],
            "answer": result["answer"],
            "found": result["found"],
            "task_type": result["task_type"],
            "contexts": result["contexts"]
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}
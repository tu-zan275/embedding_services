from fastapi import FastAPI
from rag_service import rag_answer
from milvus_client import insert_course_chunks, create_collection
from models.requests.InsertPayload import InsertPayload

app = FastAPI()

# Tạo collection khi start
create_collection()

@app.get("/")
def root():
    return {"status": "ok", "message": "RAG API is running"}

@app.post("/insert")
def insert(payload: InsertPayload):
    """
    Insert nhiều chunks vào course.
    Ví dụ payload JSON:
    {
        "course_id": 1,
        "chunks": ["Nội dung 1", "Nội dung 2"]
    }
    """
    try:
        course_id = payload.course_id
        chunks = payload.chunks
        insert_course_chunks(course_id, chunks)
        return {"status": "ok", "course_id": course_id, "chunks_count": len(chunks)}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.post("/ask")
def ask(query: str):
    answer = rag_answer(query)
    return answer

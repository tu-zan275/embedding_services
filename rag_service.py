from pymilvus import Collection
from embedding import get_embedding
from openai import OpenAI
import os
from dotenv import load_dotenv
from modules.course_rag_pipeline import query_rag  # ✅ import lại hàm

#from special_contexts import special_contexts

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def rag_answer_v1(query: str, top_k=3):
    # load collection
    collection = Collection("course_chunks")
    q_emb = get_embedding(query, is_query=True)

    # search in Milvus
    results = collection.search(
        [q_emb],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["course_id", "chunk_index", "text"]
    )

    # build context
    contexts = []
    for hit in results[0]:
        course_name = hit.entity.get("course_name")  # metadata
        course_id = hit.entity.get("course_id")
        chunk_index = hit.entity.get("chunk_index")
        text = hit.entity.get("text")

        # Tạo context rõ ràng cho AI
        contexts.append(
            f"Khóa học: {course_name} (ID: {course_id})\nChunk {chunk_index}: {text}"
        )

    context_text = "\n".join(contexts)

    #context_text += special_contexts(query)

    prompt = f"""Bạn là trợ lý AI cho hệ thống khóa học.
Người dùng hỏi: "{query}"

Dựa trên dữ liệu trong các khóa học:
{context_text}
    
Hãy trả lời ngắn gọn, chính xác, và chỉ dựa vào thông tin trên.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # hoặc model local
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "query": query,
        "contexts": contexts,
        "answer": response.choices[0].message.content
    }

def rag_answer_v2(query: str, top_k=3):
    # 1️⃣ Lấy dữ liệu từ Milvus
    collection = Collection("course_rag")

    # 2️⃣ Gọi query_rag() để tìm kết quả semantic
    results = query_rag(collection, query, limit=top_k)

    # 3️⃣ Nếu không có kết quả → fallback
    if not results or not results[0]:
        fallback_prompt = f"""
        Người dùng hỏi: "{query}".
        Tôi không tìm thấy thông tin nào liên quan trong cơ sở dữ liệu khóa học.
        Hãy trả lời lịch sự và gợi ý người dùng thử câu hỏi khác.
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": fallback_prompt}]
        )
        return {
            "query": query,
            "contexts": [],
            "answer": response.choices[0].message.content,
            "found": False
        }

    # 4️⃣ Chuẩn bị context từ kết quả search
    contexts = []
    for hit in results[0]:
        item_type = hit.entity.get("type") or ""
        course_title = hit.entity.get("course_title") or ""
        lesson_title = hit.entity.get("lesson_title") or ""
        author = hit.entity.get("author") or ""
        url = hit.entity.get("url") or ""
        content = hit.entity.get("content") or ""

        ctx = f"[{item_type.upper()}] {course_title}"
        if lesson_title:
            ctx += f" → {lesson_title}"
        ctx += f"\nTác giả: {author}\nURL: {url}\nNội dung: {content}"
        contexts.append(ctx)

    context_text = "\n\n".join(contexts)[:4000]  # tránh quá dài

    # 5️⃣ Nhận diện intent đơn giản
    query_lower = query.lower()
    if "giá" in query_lower or "bao nhiêu" in query_lower:
        task_type = "price"
    elif "tác giả" in query_lower or "ai dạy" in query_lower:
        task_type = "author"
    elif "bài học" in query_lower or "nội dung" in query_lower:
        task_type = "lessons"
    else:
        task_type = "general"

    # 6️⃣ Sinh prompt chính
    prompt = f"""
    Bạn là trợ lý AI của hệ thống khóa học.
    Người dùng hỏi: "{query}"

    Dựa trên dữ liệu được tìm thấy:
    {context_text}

    Nhiệm vụ: Trả lời ngắn gọn, rõ ràng, chỉ dựa vào thông tin trên.
    """

    if task_type == "price":
        prompt += "\nNếu có thông tin về giá, hãy nêu cụ thể. Nếu không, nói 'không có thông tin về giá'."
    elif task_type == "author":
        prompt += "\nNếu có tác giả, hãy nêu rõ ai là người dạy khóa học."
    elif task_type == "lessons":
        prompt += "\nNếu có danh sách bài học, hãy tóm tắt số lượng và tiêu đề các bài học chính."

    # 7️⃣ Gọi OpenAI để sinh câu trả lời
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "query": query,
        "contexts": contexts,
        "answer": response.choices[0].message.content,
        "found": True,
        "task_type": task_type
    }
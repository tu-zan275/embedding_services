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

def rag_answer_v2(query: str, top_k=10):
    collection = Collection("course_rag")

    preprocessed = preprocess_query_with_llm(query)

    query_clean = preprocessed["query"]
    search_type = preprocessed["type"]

    print(f"llmQuery: {query_clean} | Type: {search_type}")

    # Semantic search
    results = query_rag(collection, query_clean, limit=top_k, input_search_type=search_type)

    results = [r for r in results if r.get("score", 0) > 0.25]

    # Fallback khi không tìm thấy gì
    if not results:
        fallback_prompt = f"""
        Người dùng hỏi: "{query}".
        Tôi không tìm thấy thông tin nào trong cơ sở dữ liệu khóa học.
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

    # Chuẩn bị context
    contexts = []
    for hit in results:
        ctx = f"[{hit['type'].upper()}] {hit['course_title']}"
        if hit["lesson_title"]:
            ctx += f" → {hit['lesson_title']}"
        ctx += f"\nTác giả: {hit['author']}\nURL: {hit['url']}\nNội dung: {hit['content']}"
        contexts.append(ctx)

    context_text = "\n\n".join(contexts)[:4000]

    # 4️⃣ Nhận diện intent đơn giản
    query_lower = query.lower()
    if "giá" in query_lower or "bao nhiêu" in query_lower:
        task_type = "price"
    elif "tác giả" in query_lower or "ai dạy" in query_lower:
        task_type = "author"
    elif "bài học" in query_lower or "nội dung" in query_lower:
        task_type = "lessons"
    else:
        task_type = "general"

    # 5️⃣ Sinh prompt chính
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

    # 6️⃣ Gọi OpenAI để sinh câu trả lời
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

def rag_search(query: str, top_k=10):
    collection = Collection("course_rag")

    # Semantic search
    results = query_rag(collection, query, limit=top_k)

    results = [r for r in results if r.get("score", 0) > 0.25]

    # Fallback khi không tìm thấy gì
    if not results:
        return {
            "query": query,
            "results": [],
            "found": False
        }

    return {
        "query": query,
        "results": results,
        "found": True,
    }


def preprocess_query_with_llm(query: str) -> dict:
    """
    Chuẩn hóa query và phân loại type ('course' hoặc 'lesson') bằng LLM.
    Trả về dict: {"query": normalized_query, "type": "course"|"lesson"}
    """
    prompt = f"""
    Bạn là một trợ lý AI. 
    1. Viết lại câu hỏi sau cho ngắn gọn, chuẩn hóa, loại bỏ các từ dư thừa nhưng vẫn giữ nguyên ý nghĩa.
    2. Phân loại câu hỏi này thành 'course' (tổng quan khóa học) hoặc 'lesson' (câu hỏi chi tiết bài học).
    
    Trả về JSON duy nhất với 2 key: 
    {{
        "query": "câu hỏi đã chuẩn hóa",
        "type": "course" hoặc "lesson"
    }}
    
    Câu hỏi: "{query}"
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    try:
        import json
        json_text = response.choices[0].message.content.strip()
        result = json.loads(json_text)
        # fallback kiểm tra type
        if result.get("type") not in ["course", "lesson"]:
            result["type"] = "course"
        return result
    except Exception as e:
        # fallback nếu JSON lỗi
        return {"query": query, "type": "course"}

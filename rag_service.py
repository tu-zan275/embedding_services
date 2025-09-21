from pymilvus import Collection
from embedding import get_embedding
from openai import OpenAI
import os
from dotenv import load_dotenv

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def rag_answer(query: str, top_k=3):
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
        contexts.append(
            f"(Course {hit.entity.get('course_id')}, Chunk {hit.entity.get('chunk_index')}): {hit.entity.get('text')}"
        )

    context_text = "\n".join(contexts)

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

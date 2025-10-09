LASTEST

course_rag_pipeline.py
main.py
AskPayload.py
InsertPayload.py
rag_service.py











========================================================================================================









👉 Nếu bạn định chạy embedding local (dùng sentence-transformers) thay vì OpenAI API, thì chỉ cần thêm:

sentence-transformers==2.7.0
torch>=2.2.0


Nếu muốn nhanh, nhẹ, dễ chạy CPU → intfloat/multilingual-e5-small.
Nếu muốn chính xác hơn, dataset quan trọng → intfloat/multilingual-e5-base.
Nếu chỉ dùng tiếng Việt → keepitreal/vietnamese-sbert.
Nếu muốn đa ngôn ngữ mạnh nhất (nhưng nặng) → bge-m3.


curl -X POST "http://127.0.0.1:8000/insert" -H "Content-Type: application/json" -d '{"text": "Khóa học Python cơ bản cho người mới bắt đầu"}'

curl -X POST "http://127.0.0.1:8000/query" -H "Content-Type: application/json" -d '{"text": "Tôi muốn học lập trình Python", "top_k": 2}'


===========================
📌 Cách dùng
chmod +x install_milvus.sh
./install_milvus.sh


Sau khi chạy xong, kiểm tra container:
docker ps

Kết quả sẽ có milvus-standalone, milvus-minio, milvus-etcd.

===========================
wsl
# Tạo folder cho venv (ví dụ tên venv)
python3 -m venv venv

# Kích hoạt virtualenv
source venv/bin/activate  # Linux / WSL

# Kiểm tra pip đang dùng venv
which pip

# Cài các package từ requirements.txt
pip install -r requirements.txt

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load biến môi trường từ .env
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Tạo client mới
client = OpenAI(api_key=OPENAI_API_KEY)

def test_openai_key():
    """
    Kiểm tra API key có hợp lệ không.
    Trả về True nếu hợp lệ, False nếu lỗi.
    """
    try:
        # Gọi endpoint simple để test, ví dụ list models
        models = client.models.list()
        print("✅ OpenAI API key is valid. Models count:", len(models.data))
        return True
    except Exception as e:
        print("❌ Error:", str(e))
        return False

# =====================
# Test
# =====================
if __name__ == "__main__":
    valid = test_openai_key()
    print("Key valid:", valid)

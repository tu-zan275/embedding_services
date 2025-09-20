from sentence_transformers import SentenceTransformer

# Load model (chỉ load 1 lần khi khởi động server)
model = SentenceTransformer("intfloat/multilingual-e5-small")

def get_embedding(text: str, is_query: bool = False):
    """
    Sinh embedding cho text.
    :param text: câu input
    :param is_query: True nếu là query, False nếu là passage/document
    :return: list float (vector)
    """
    if is_query:
        text = "query: " + text
    else:
        text = "passage: " + text

    return model.encode(text, normalize_embeddings=True).tolist()

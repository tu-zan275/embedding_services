from pydantic import BaseModel

class AskPayload(BaseModel):
    query: str
    top_k: int = 3
from pydantic import BaseModel

class AskPayload(BaseModel):
    query: str
#InsertPayload.py
from pydantic import BaseModel
from typing import List

class InsertPayload(BaseModel):
    course_id: int
    chunks: List[str]
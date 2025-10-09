# models/requests/payloads.py
from pydantic import BaseModel
from typing import List, Optional

class Lesson(BaseModel):
    lesson_id: str
    title: str
    content: str

class InsertPayload(BaseModel):
    course_id: str
    title: str
    author: str
    category: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None
    lessons: List[Lesson]

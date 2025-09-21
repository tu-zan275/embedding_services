# system_api.py
from typing import Dict, Any
import requests

class SystemAPI:
    """
    Class này quản lý các call API nội bộ để lấy thông tin hệ thống.
    Ví dụ: tổng số khóa học, chi tiết khóa học, khóa học mới nhất...
    """

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")  # đảm bảo không có / cuối

    def get_total_courses(self) -> int:
        """
        Lấy tổng số khóa học trong hệ thống.
        """
        url = f"{self.base_url}/courses/count"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return data.get("total", 0)

    def get_latest_course(self) -> Dict[str, Any]:
        """
        Lấy thông tin khóa học mới nhất.
        """
        url = f"{self.base_url}/courses/latest"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return data

    def get_course_detail(self, course_id: int) -> Dict[str, Any]:
        """
        Lấy chi tiết một khóa học theo ID.
        """
        url = f"{self.base_url}/courses/{course_id}"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return data

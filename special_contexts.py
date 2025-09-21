from system_api import SystemAPI

api = SystemAPI("http://localhost:8000")  # hoặc URL hệ thống của bạn

def special_contexts(query: str) -> str:
    """
    Trả về context đặc biệt dựa trên query.
    Nếu query không thỏa điều kiện nào → trả về "".
    """
    query_lower = query.lower()  # chuẩn hóa để so sánh

    # Case 1: hỏi tổng số khóa học
    if "tổng cộng" in query_lower or "bao nhiêu khóa học" in query_lower:
        total = api.get_total_courses()
        print("Tổng số khóa học:", total)
        return f"Hệ thống hiện có tổng cộng {total_courses} khóa học."

    # Case 2: hỏi khóa học mới nhất
    if "khóa học mới nhất" in query_lower or "recent course" in query_lower:
        latest_course = get_latest_course()  # gọi API/DB
        return f"Khóa học mới nhất là '{latest_course['name']}' (ID: {latest_course['id']})."

    # Case 3: các case khác có thể thêm ở đây
    # elif "từ khóa khác" in query_lower:
    #     ...

    # Nếu không thỏa điều kiện nào
    return ""


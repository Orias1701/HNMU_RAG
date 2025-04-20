import json
import re
from typing import Dict, List
import google.generativeai as genai

def generate_queries_with_gemini(text: str, api_key: str, n_queries: int = 3) -> List[str]:
    """Tạo truy vấn giả bằng Gemini API."""
    # Cấu hình API key
    genai.configure(api_key=api_key)
    
    # Chọn mô hình Gemini (giả định gemini-1.5-flash hoặc tương tự)
    model = genai.GenerativeModel("gemini-2.0-flash-exp")  # Thay bằng "gemini-flash-2.0-experimental" nếu có
    
    # Lời nhắc để tạo câu hỏi
    prompt = (
        f"Đọc văn bản pháp lý sau và tạo {n_queries} câu hỏi liên quan, ngắn gọn, tự nhiên, "
        f"phù hợp với ngữ cảnh pháp lý tiếng Việt. Tránh lặp lại tiêu đề hoặc từ khóa chung như 'phạm vi', 'đối tượng'. "
        f"Câu hỏi nên giống cách người dùng thực tế hỏi. Mỗi câu hỏi trên một dòng:\n\n{text[:500]}\n\nCâu hỏi:"
    )
    
    try:
        response = model.generate_content(prompt, generation_config={"max_output_tokens": 100, "temperature": 0.7})
        queries = response.text.strip().split('\n')[:n_queries]
        return [q.strip() for q in queries if q.strip()]
    except Exception as e:
        print(f"Lỗi khi gọi Gemini API: {str(e)}")
        return []

def generate_synthetic_ground_truth(
    mapping_data: str,
    mapping_path: str,
    output_path: str,
    gemini_api_key: str,
    max_queries: int = 1000
) -> None:
    """Tạo ground truth với truy vấn giả sử dụng Gemini API."""
    with open(mapping_data, 'r', encoding='utf-8') as f:
        data_mapping = json.load(f)
    with open(mapping_path, 'r', encoding='utf-8') as f:
        key_to_index = json.load(f)
    
    ground_truth = {}
    query_count = 0
    
    for key, text in data_mapping.items():
        if "Merged_text" not in key:
            continue
        
        # Tạo truy vấn giả bằng Gemini API
        queries = generate_queries_with_gemini(text, gemini_api_key, n_queries=3)
        
        # Dự phòng: Tạo câu hỏi từ tiêu đề/dòng đầu tiên nếu API thất bại
        if not queries:
            sentences = re.split(r'[.!?]', text)
            query = sentences[0].strip()[:100]
            queries = [f"Nội dung của {query} là gì?"] if query else []
        
        embedding_key = key.replace("Merged_text", "Merged_embedding")
        if embedding_key not in key_to_index:
            continue
        
        # Thêm các truy vấn vào ground truth
        for query in queries:
            ground_truth[query] = [embedding_key]
            query_count += 1
            if query_count >= max_queries:
                break
        if query_count >= max_queries:
            break
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, ensure_ascii=False, indent=2)
    
    print(f"Đã tạo ground truth với {query_count} truy vấn, lưu tại: {output_path}")

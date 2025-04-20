import json
import re
import os
from typing import List, Dict, Any
from underthesea import word_tokenize
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

""" RESPOND """

"""
Lọc kết quả rerank và sinh câu trả lời tự nhiên bằng Gemini 1.5 Pro.

Args:
    query: Câu hỏi dạng văn bản
    results: Danh sách kết quả từ rerank_results ({'text', 'rerank_score', 'faiss_score', 'key'})
    responser_model: Tên mô hình Gemini (mặc định gemini-2.0-flash-exp)
    device: Thiết bị PyTorch (cuda hoặc cpu, chỉ để tương thích)
    score_threshold: Ngưỡng rerank_score để lọc
    max_results: Số kết quả tối đa để tổng hợp
    gemini_api_key: API key của Google AI Studio

Returns:
    Tuple: (câu trả lời tự nhiên, danh sách kết quả được lọc)
"""

def respond_naturally(
    # prompt,
    user_question: str,
    results: List[Dict[str, Any]],
    system_prompt: List[Dict[str, Any]],
    responser_model: str = "gemini-2.0-flash-exp",
    score_threshold: float = 0.85,
    max_results: int = 3,
    doc: bool = True,
    gemini_api_key: str = None,
) -> tuple[str, List[Dict[str, Any]]]:
    
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel(responser_model)

        if (doc):
            # Sort kết quả
            filtered_results = [
                r for r in results
                if r["rerank_score"] > score_threshold and len(r["text"]) > 50
            ][:max_results]\
            
            context = "\n".join([r["text"] for r in filtered_results])      
            prompt = (
                f"{system_prompt} \n"
                f"Tài liệu: {context} \n \n"
                f"Trả lời cầu hỏi của tôi: {user_question}"
            )
        else:
            prompt = (
                f"{system_prompt} \n"
                f"Trả lời cầu hỏi của tôi: {user_question}"
            )
        
        # Sinh câu trả lời
        response = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": 512,
                "temperature": 0.3,
                "top_p": 0.9
            }
        )
        
        # Xử lý response
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and candidate.content.parts:
                response_text = candidate.content.parts[0].text.strip()
            else:
                raise ValueError("Không tìm thấy nội dung trong candidate của Gemini API.")
        else:
            raise ValueError("Response không có candidates.")

        return response_text, filtered_results
    
    except ResourceExhausted as e:
        error_msg = f"Vượt giới hạn API"
        print(error_msg)
        return ("Vượt giới hạn API, vui lòng thử lại sau.", [])
def generate_queries_with_gemini(text: str, api_key: str, n_queries: int = 3) -> List[str]:
    """Tạo truy vấn giả bằng Gemini API."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    
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
        print(f"Lỗi khi tạo câu hỏi: {str(e)}")
        return []

# Hàm tạo qa_pairs từ results
def generate_qa_pairs_from_results(
    results: List[Dict[str, Any]],
    qa_pairs_path: str,
    gemini_api_key: str,
    system_prompt: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, str]]]:
    """Tạo câu hỏi và câu trả lời tham chiếu từ results."""
    if os.path.exists(qa_pairs_path):
        with open(qa_pairs_path, 'r', encoding='utf-8') as f:
            qa_pairs = json.load(f)
            # Kiểm tra xem tất cả results đã có trong qa_pairs chưa
            missing_keys = [r["key"] for r in results if r["key"] not in qa_pairs]
            if not missing_keys:
                return qa_pairs
    
    qa_pairs = {} if not os.path.exists(qa_pairs_path) else json.load(open(qa_pairs_path, 'r', encoding='utf-8'))
    
    for result in results:
        embedding_key = result["key"]
        text = result["text"]
        
        if embedding_key in qa_pairs:
            continue
        
        # Tạo câu hỏi
        queries = generate_queries_with_gemini(text, gemini_api_key, n_queries=3)
        if not queries:
            sentences = re.split(r'[.!?]', text)
            query = sentences[0].strip()[:100]
            queries = [f"Nội dung của {query} là gì?"] if query else []
        
        # Tạo câu trả lời tham chiếu
        qa_list = []
        for query in queries:
            single_result = [{"text": text, "rerank_score": 1.0, "key": embedding_key}]
            response, _ = respond_naturally(
                user_question=query,
                results=single_result,
                system_prompt=system_prompt,
                responser_model="gemini-2.0-flash-exp",
                score_threshold=0.85,
                max_results=3,
                doc=True,
                gemini_api_key=gemini_api_key
            )
            if "Lỗi" not in response:
                qa_list.append({"question": query, "answer": response})
        
        if qa_list:
            qa_pairs[embedding_key] = qa_list
    
    with open(qa_pairs_path, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    
    print(f"Đã tạo/cập nhật {len(qa_pairs)} cặp QA, lưu tại: {qa_pairs_path}")
    return qa_pairs

# Hàm tính F1 score
def calculate_f1_score(predicted_text: str, reference_text: str, stop_words: set = None) -> float:
    """Tính điểm F1 giữa câu trả lời dự đoán và tham chiếu."""
    if stop_words is None:
        stop_words = {"là", "của", "và", "trong", "được", "các", "này", "thì"}

    pred_tokens = word_tokenize(predicted_text.lower())
    ref_tokens = word_tokenize(reference_text.lower())
    
    pred_tokens = [t for t in pred_tokens if t not in stop_words]
    ref_tokens = [t for t in ref_tokens if t not in stop_words]
    
    common_tokens = set(pred_tokens) & set(ref_tokens)
    
    precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(common_tokens) / len(ref_tokens) if ref_tokens else 0.0
    
    if precision + recall == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
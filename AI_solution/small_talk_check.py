import unicodedata
from rapidfuzz import fuzz


def to_lower_case(text):
    text = text.lower()
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    return text


def small_talk_check(sentence, threshold=80):
    clean_text = to_lower_case(sentence)
    
    
    small_talk_keywords = [
        "xin chao", "hom nay", "khoe khong", "the nao", "cam thay", "noi chuyen", "vui ve", "tro chuyen",
        "toi ten la", "ten toi la", "toi o dau", "ban o dau", "co met khong", "ban nghi sao", "anh nghi sao",
        "cam on", "xin loi", "toi buon", "toi vui", "troi dep", "troi mua", "co bi khong", "co that khong",
        "co dung khong", "thay co tot khong", "cai nay sao", "ban thay sao", "co bip hay khong", "dung that khong",
        "thay dạy như cứt", "dạy như thế nào", "thầy dạy môn này sao", "thầy dạy môn này chán", "môn này như cứt",
        "co dạy như thế nào", "co ABC", "co dạy môn này sao", "co dạy môn này chán", "co như thế nào", "co bip hay khong",
        "tệ", "chán", "xấu", "như cứt", "quá tệ", "cực kỳ tệ", "không hay", "chán ngắt", "cái này chán", "dở",
        "có gì mới không", "bạn nghĩ sao", "bạn có khỏe không", "bạn làm gì thế", "bạn thích ăn gì", "tôi buồn", "tôi vui",
        "tôi mệt", "thật tệ", "rất tốt", "cảm ơn", "cảm ơn bạn", "xin lỗi", "chào bạn", "hôm nay bạn thế nào",
        "bạn đang làm gì", "thầy giảng như thế nào", "môn này có khó không", "môn này như thế nào", "môn học này có gì hay",
        "có thầy cô nào dạy tốt không", "thầy dạy như thế nào", "co bip hay khong", "dạy như thế nào", "hello", "hi"
    ]
    
    
    courses_info_keywords = [
        "ma mon hoc", "mon hoc", "sinh vien", "noi dung", "kien thuc", "giang day", "phuong phap", 
        "bai tap", "thoi khoa bieu", "ky thi", "dai hoc", "giang duong", "bai giang", "hoc ky", 
        "tai lieu", "truong hoc", "ky nang", "ky thuat", "thuc hanh", "kiem tra", "de thi", "sach giao khoa"
    ]
    
    # Kiểm tra mức độ khớp với small talk
    small_talk_match = any(fuzz.partial_ratio(clean_text, kw) >= threshold for kw in small_talk_keywords)
    
    # Kiểm tra mức độ khớp với thông tin môn học
    courses_info_match = any(fuzz.partial_ratio(clean_text, kw) >= threshold for kw in courses_info_keywords)
    
    
    return 1 if small_talk_match and not courses_info_match else 0


if __name__ == "__main__":
    test_sentences = [
        "Tôi tên là Tài",  
        "Môn CS231 là môn gì",  
        "Cho tôi xin thông tin về môn cs116",  
        "Thầy Lê Đình Duy dạy môn giới thiệu ngành khoa học máy tính có bịp hay không",  
        "Xin chào, hôm nay bạn thế nào?",  
        "Môn học Dịch máy có mã môn học là CS325.",  
        "Môn xử lý ngôn ngữ tự nhiên là học cái gì?", 
        "Bạn có khoẻ không?", 
        "Thầy giảng bài có hay không?",  
        "Môn học này rất hay và cung cấp nhiều kiến thức hữu ích.",  
        "Thầy ABC dạy môn CE231 như cứt",
        "Bạn cho tôi xin thông tin thêm và giảng viên dạy môn giáo dục thể chất đi",  
        "Cô Diễm dạy DSA có hay hay không?",  
        "Môn học này quá tệ",  
        "Môn học này cực kỳ tệ",  
    ]
    
    for sentence in test_sentences:
        print(f"Input: {sentence}")
        print(f"Small Talk: {small_talk_check(sentence)}\n")

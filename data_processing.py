# -*- coding: utf-8 -*-
"""
=============================================================================
ĐỒ ÁN: KHAI THÁC DỮ LIỆU & TRUYỀN THÔNG XÃ HỘI E-COMMERCE SHOPEE
MODULE: TIỀN XỬ LÝ DỮ LIỆU & TỪ ĐIỂN TEENCODE (data_processing.py)
NHÓM THỰC HIỆN:
- Trần Đình Huy (MSSV: 24730103) - Nhóm trưởng
- Lê Thanh Trúc Vi (MSSV: 24730150) - Phụ trách Tiền xử lý NLP & Teencode
- Vũ Hoàng Thiên Ân (MSSV: 24730155) - Phụ trách Gom cụm & Dữ liệu
- Dương Phương Anh (MSSV: 24730156) - Phụ trách Đánh giá & Giao diện
=============================================================================
"""

import os
import sys
import re
import json
import hashlib
import random
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Thử nạp underthesea nếu có, fallback sang regex word tokenizer nếu chưa tải mô hình
try:
    from underthesea import word_tokenize
    HAS_UNDERTHESEA = True
except Exception:
    HAS_UNDERTHESEA = False

# =============================================================================
# 1. TỪ ĐIỂN CHUẨN HÓA TEENCODE & THUẬT NGỮ THƯƠNG MẠI ĐIỆN TỬ
# =============================================================================
TEENCODE_DICT = {
    # Viết tắt thông dụng
    "sp": "sản phẩm",
    "spm": "sản phẩm mới",
    "shop": "cửa hàng",
    "sop": "cửa hàng",
    "shopp": "cửa hàng",
    "đc": "được",
    "dc": "được",
    "dk": "được",
    "dok": "được",
    "k": "không",
    "ko": "không",
    "kh": "không",
    "khong": "không",
    "hok": "không",
    "hong": "không",
    "hông": "không",
    "kp": "không phải",
    "kô": "không",
    "ntn": "như thế này",
    "j": "gì",
    "gi": "gì",
    "cx": "cũng",
    "cug": "cũng",
    "thui": "thôi",
    "wa": "quá",
    "qua": "quá",
    "rat": "rất",
    "nhju": "nhiều",
    "nhieu": "nhiều",
    "nhìu": "nhiều",
    "vs": "với",
    "v": "vậy",
    "vay": "vậy",
    "uk": "ừ",
    "uh": "ừ",
    "tks": "cảm ơn",
    "thx": "cảm ơn",
    "cảm ơn": "cảm ơn",
    "cam on": "cảm ơn",
    "tks shop": "cảm ơn cửa hàng",
    "thks": "cảm ơn",
    "ty": "cảm ơn",
    "tks u": "cảm ơn bạn",
    "gud": "tốt",
    "good": "tốt",
    "ok": "tốt",
    "oki": "tốt",
    "okie": "tốt",
    "oke": "tốt",
    "okey": "tốt",
    "okela": "tốt",
    "oklah": "tốt",
    "tuyet": "tuyệt",
    "tuyet voi": "tuyệt vời",
    "tuyệt vời": "tuyệt vời",
    "chat luong": "chất lượng",
    "đẹp": "đẹp",
    "dep": "đẹp",
    "xịn": "xịn sò",
    "xin": "xịn",
    "mlem": "ngon đẹp",
    "ưng": "hài lòng",
    "ung": "hài lòng",
    "ưng bụng": "rất hài lòng",
    "iu": "yêu thích",
    "love": "yêu thích",
    "thik": "thích",
    "me": "mê",
    
    # Từ ngữ E-commerce & Vận chuyển
    "ship": "giao hàng",
    "shipper": "người giao hàng",
    "giao hang": "giao hàng",
    "gh": "giao hàng",
    "nhanh": "nhanh",
    "toc bien": "rất nhanh",
    "hỏa tốc": "giao hỏa tốc",
    "cham": "chậm",
    "rùa bò": "rất chậm",
    "lau": "lâu",
    "dong goi": "đóng gói",
    "đg": "đóng gói",
    "bọc": "đóng gói",
    "can than": "cẩn thận",
    "chắc chắn": "chắc chắn",
    "sơ sài": "kém cẩn thận",
    "móp méo": "hư hại móp méo",
    "vỡ": "hư hỏng",
    "hỏng": "hư hỏng",
    "rách": "rách hỏng",
    "auth": "hàng chính hãng",
    "authentic": "hàng chính hãng",
    "chính hãng": "hàng chính hãng",
    "fake": "hàng giả",
    "pha ke": "hàng giả",
    "rep 1:1": "hàng nhái",
    "rep": "hàng nhái",
    "dỏm": "kém chất lượng",
    "lởm": "kém chất lượng",
    "dở": "tệ",
    "tệ": "tệ",
    "te": "tệ",
    "fail": "thất vọng",
    "that vong": "thất vọng",
    "lừa đảo": "gian lận",
    "lua dao": "gian lận",
    "scam": "gian lận",
    "boom": "bùng hàng hủy đơn",
    "hoàn tiền": "yêu cầu trả hàng hoàn tiền",
    "doi tra": "đổi trả",
    "voucher": "mã giảm giá",
    "magiamgia": "mã giảm giá",
    "sale": "khuyến mãi",
    "sale off": "giảm giá",
    "flashsale": "giảm giá chớp nhoáng",
    "freeship": "miễn phí vận chuyển",
    "fs": "miễn phí vận chuyển",
    "đắt": "giá cao",
    "mắc": "giá cao",
    "rẻ": "giá rẻ hợp lý",
    "re": "giá rẻ hợp lý",
    "hạt dẻ": "giá rất rẻ",
    "đáng tiền": "đáng giá tiền",
    "tiền nào của nấy": "chất lượng tương xứng giá",
    "tu van": "tư vấn hỗ trợ",
    "rep ib": "trả lời tin nhắn",
    "rep inbox": "trả lời tin nhắn",
    "tl": "trả lời",
    "nhiet tinh": "nhiệt tình",
    "dễ thương": "thân thiện nhiệt tình",
    "cute": "dễ thương thân thiện",
    "thô lỗ": "thái độ kém",
    "chảnh": "thái độ kém",
    "ko rep": "không trả lời chăm sóc kém"
}

# Danh sách Emoji tích cực & tiêu cực
EMOJI_SENTIMENT = {
    "❤️": " rất yêu thích ", "😍": " cực kỳ ưng ý ", "🥰": " rất hài lòng ",
    "👍": " chất lượng tốt ", "💯": " hoàn hảo xuất sắc ", "🔥": " rất tuyệt vời ",
    "⭐": " năm sao ", "⭐⭐⭐⭐⭐": " đánh giá năm sao tuyệt đối ",
    "😭": " thất vọng buồn ", "😡": " rất tức giận ", "🤬": " bức xúc phẫn nộ ",
    "👎": " rất tệ không hài lòng ", "🤮": " quá dở tệ hại ", "💔": " thất vọng hoàn toàn ",
    "📦": " bưu kiện đóng gói ", "🚚": " vận chuyển giao hàng "
}

# Stopwords tiếng Việt cơ bản cho E-commerce
VIETNAMESE_STOPWORDS = {
    "và", "của", "cho", "ở", "tại", "thì", "là", "được", "với", "những",
    "các", "một", "này", "đó", "khi", "trong", "trên", "ra", "đã", "sẽ",
    "vào", "lại", "đến", "bởi", "do", "về", "như", "đang", "từ"
}

# =============================================================================
# 2. CÁC HÀM TIỀN XỬ LÝ VĂN BẢN (TEXT PREPROCESSING PIPELINE)
# =============================================================================

def clean_vietnamese_text(text: str) -> str:
    """
    Tiền xử lý và làm sạch văn bản đánh giá / bình luận tiếng Việt:
    1. Chuyển chữ thường
    2. Chuyển đổi emoji thành từ ngữ cảm xúc
    3. Chuẩn hóa URL, email, ký tự đặc biệt rác
    4. Thay thế teencode theo từ điển Thương mại điện tử
    5. Xóa ký tự lặp vô nghĩa (vd: đẹpppppp -> đẹp, okiêeeeee -> oke)
    6. Tách từ tiếng Việt
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # 1. Chuyển chữ thường
    text = text.lower()
    
    # 2. Xử lý Emoji
    for emoji_char, replacement in EMOJI_SENTIMENT.items():
        text = text.replace(emoji_char, replacement)
    
    # 3. Chuẩn hóa liên kết / mã đơn / email
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' ', text)
    
    # 4. Rút gọn ký tự lặp kéo dài (vd: ngonn quáaaaa -> ngon quá)
    text = re.sub(r'([a-zà-ỹ])\1{2,}', r'\1', text)
    
    # 5. Xóa các ký tự không phải chữ cái tiếng Việt hoặc số
    text = re.sub(r'[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', ' ', text)
    
    # 6. Thay thế Teencode theo từng từ
    words = text.split()
    normalized_words = []
    i = 0
    while i < len(words):
        # Kiểm tra cụm 2 từ trước
        if i + 1 < len(words):
            two_words = f"{words[i]} {words[i+1]}"
            if two_words in TEENCODE_DICT:
                normalized_words.append(TEENCODE_DICT[two_words])
                i += 2
                continue
        # Kiểm tra từ đơn
        w = words[i]
        if w in TEENCODE_DICT:
            normalized_words.append(TEENCODE_DICT[w])
        else:
            normalized_words.append(w)
        i += 1
    
    processed_text = " ".join(normalized_words)
    
    # 7. Tách từ tiếng Việt (Word Segmentation)
    if HAS_UNDERTHESEA:
        try:
            processed_text = word_tokenize(processed_text, format="text")
        except Exception:
            pass
            
    # Chuẩn hóa khoảng trắng dư thừa
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    return processed_text


def calculate_lexicon_sentiment_score(text: str) -> Tuple[float, str, List[str], List[str]]:
    """
    Tính điểm cảm xúc dựa trên Lexicon E-Commerce Tiếng Việt kết hợp quy tắc phủ định.
    Trả về:
    - score: float từ -1.0 (cực kỳ tiêu cực) đến +1.0 (cực kỳ tích cực)
    - label: 'Tích cực' | 'Trung lập' | 'Tiêu cực'
    - pos_keywords: Danh sách từ khóa tích cực phát hiện
    - neg_keywords: Danh sách từ khóa tiêu cực phát hiện
    """
    cleaned = clean_vietnamese_text(text)
    
    POS_WORDS = {
        "tốt": 1.0, "tuyệt": 1.2, "tuyệt_vời": 1.5, "ưng_ý": 1.2, "hài_lòng": 1.2,
        "xịn": 1.0, "xịn_sò": 1.2, "chất_lượng": 1.0, "đẹp": 1.0, "rẻ": 0.8,
        "hợp_lý": 0.8, "đáng_tiền": 1.0, "nhanh": 0.9, "cẩn_thận": 0.9, "chính_hãng": 1.2,
        "nhiệt_tình": 1.0, "thân_thiện": 1.0, "yêu_thích": 1.2, "hoàn_hảo": 1.5,
        "xuất_sắc": 1.5, "năm_sao": 1.2, "chắc_chắn": 0.8, "cảm_ơn": 0.6, "đúng_mô_tả": 1.2
    }
    
    NEG_WORDS = {
        "tệ": -1.2, "dở": -1.0, "xấu": -1.0, "kém": -1.0, "dỏm": -1.3,
        "lởm": -1.3, "hư_hỏng": -1.5, "vỡ": -1.4, "rách": -1.4, "chậm": -0.9,
        "rùa_bò": -1.2, "lâu": -0.8, "móp_méo": -1.1, "kém_cẩn_thận": -1.0,
        "hàng_giả": -1.8, "hàng_nhái": -1.5, "thất_vọng": -1.4, "gian_lận": -2.0,
        "lừa_đảo": -2.0, "thái_độ_kém": -1.3, "chăm_sóc_kém": -1.1, "đắt": -0.7,
        "giá_cao": -0.6, "bùng_hàng": -1.5, "sai_hàng": -1.4, "thiếu_hàng": -1.3
    }
    
    NEGATION_WORDS = {"không", "chẳng", "chả", "không_phải", "kém", "chưa"}
    
    words = cleaned.split()
    pos_found = []
    neg_found = []
    total_score = 0.0
    
    i = 0
    while i < len(words):
        w = words[i]
        is_negated = False
        if i > 0 and words[i-1] in NEGATION_WORDS:
            is_negated = True
        if i > 1 and words[i-2] in NEGATION_WORDS:
            is_negated = True
            
        if w in POS_WORDS:
            val = POS_WORDS[w]
            if is_negated:
                total_score -= val * 1.2
                neg_found.append(f"không {w}")
            else:
                total_score += val
                pos_found.append(w)
        elif w in NEG_WORDS:
            val = NEG_WORDS[w]
            if is_negated:
                total_score += abs(val) * 0.8
                pos_found.append(f"không {w}")
            else:
                total_score += val
                neg_found.append(w)
        i += 1
        
    # Chuẩn hóa về [-1.0, 1.0]
    if total_score > 0:
        normalized_score = min(1.0, total_score / 3.0)
    elif total_score < 0:
        normalized_score = max(-1.0, total_score / 3.0)
    else:
        normalized_score = 0.0
        
    # Gán nhãn
    if normalized_score >= 0.2:
        label = "Tích cực"
    elif normalized_score <= -0.2:
        label = "Tiêu cực"
    else:
        label = "Trung lập"
        
    return round(normalized_score, 3), label, pos_found, neg_found


# =============================================================================
# 3. TẠO BỘ DỮ LIỆU MÔ PHỎNG SHOPEE E-COMMERCE & SOCIAL MEDIA
# =============================================================================

def generate_shopee_datasets(output_dir: str = "data") -> Dict[str, pd.DataFrame]:
    """
    API cũ bị khóa để tránh ghi đè ba CSV Kaggle bằng dữ liệu mô phỏng.

    Dùng ``import_public_data.py --download`` để dựng dữ liệu hiện hành.
    """
    raise RuntimeError(
        "Đã vô hiệu hóa generator mô phỏng. "
        "Chạy import_public_data.py --download để nhập ba bộ Kaggle đã ghim phiên bản."
    )

    # Phần triển khai cũ bên dưới chỉ được giữ để truy vết lịch sử mã nguồn và không thể chạy.
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent / output_path
    output_path.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    np.random.seed(42)
    
    categories = [
        "Thời Trang Nam/Nữ", "Điện Thoại & Phụ Kiện", "Máy Tính & Laptop",
        "Mỹ Phẩm & Làm Đẹp", "Nhà Cửa & Đời Sống", "Thiết Bị Điện Gia Dụng",
        "Mẹ & Bé", "Bách Hóa Online", "Thể Thao & Du Lịch"
    ]
    
    locations = ["Hà Nội", "TP. Hồ Chí Minh", "Đà Nẵng", "Bình Dương", "Đồng Nai", "Cần Thơ", "Nước Ngoài (Trung Quốc)"]
    
    # -------------------------------------------------------------------------
    # A. TẠO DỮ LIỆU SẢN PHẨM (1,200 records)
    # -------------------------------------------------------------------------
    product_rows = []
    
    adjectives = ["Cao Cấp", "Chính Hãng", "Thông Minh", "Hot Trend", "Đa Năng", "Chống Nước", "Thế Hệ Mới", "Giá Rẻ", "Siêu Bền", "Nhập Khẩu"]
    
    prod_names = {
        "Thời Trang Nam/Nữ": ["Áo Thun Unisex Cotton 100%", "Quần Jean Baggy Form Rộng", "Váy Nữ Dáng Xòe Vintage", "Áo Khoác Bomber Chống Gió", "Giày Sneaker Thể Thao Trắng"],
        "Điện Thoại & Phụ Kiện": ["Củ Sạc Nhanh 65W GaN Type-C", "Tai Nghe Bluetooth True Wireless Chống Ồn", "Cáp Sạc Nhanh Tự Ngắt Bọc Dù", "Ốp Lưng Chống Sốc Trong Suốt", "Kính Cường Lực 9D Full Màn"],
        "Máy Tính & Laptop": ["Chuột Không Dây Silent Siêu Nhạy", "Bàn Phím Cơ LED RGB Blue/Red Switch", "Giá Đỡ Laptop Nhôm Tản Nhiệt Gấp Gọn", "Hub Chuyển Đổi Type-C 7 in 1", "Lót Chuột Gaming Khổ Lớn 80x30cm"],
        "Mỹ Phẩm & Làm Đẹp": ["Kem Chống Nắng Kiềm Dầu SPF50+", "Serum Phục Hồi Da B5 Dưỡng Ẩm Sâu", "Sữa Rửa Mặt Dịu Nhẹ Cho Da Dầu Mụn", "Son Kem Lì Mịn Môi Lâu Trôi", "Nước Tẩy Trang Micellar Water Dịu Nhẹ"],
        "Nhà Cửa & Đời Sống": ["Bình Giữ Nhiệt Inox 304 800ml", "Đèn Bàn Học LED Chống Cận Thị", "Hộp Cơm Cắm Điện Hâm Nóng Tự Động", "Gối Cao Su Non Công Thái Học", "Bộ Cây Lau Nhà Tự Vắt 360 Độ"],
        "Thiết Bị Điện Gia Dụng": ["Nồi Chiên Không Dầu Điện Tử 6.5L", "Máy Hút Bụi Cầm Tay Không Dây Siêu Hút", "Ấm Siêu Tốc 2 Lớp Cách Nhiệt 1.8L", "Máy Xay Sinh Tố Mini Cầm Tay 6 Lưỡi", "Quạt Cây Đứng Điều Khiển Từ Xa"],
        "Mẹ & Bé": ["Bình Sữa Cổ Rộng Chống Sặc PPSU", "Tã Bỉm Dán Siêu Thấm Hút Ban Đêm", "Bộ Đồ Chơi Xếp Hình Gỗ Trí Uẩn", "Xe Đẩy Gấp Gọn Siêu Nhẹ Du Lịch", "Khăn Ướt Kháng Khuẩn Không Mùi"],
        "Bách Hóa Online": ["Cà Phê Rang Xay Nguyên Chất Đậm Vị", "Hạt Dinh Dưỡng Hạnh Nhân Óc Chó Mix", "Trà Xanh Thái Nguyên Thượng Hạng", "Bánh Quy Bơ Phô Mai Giòn Rụm", "Nước Giặt Xả Hương Nước Hoa Đậm Đặc"],
        "Thể Thao & Du Lịch": ["Thảm Tập Yoga TPE 2 Lớp Chống Trượt", "Bình Nước Thể Thao 2 Lít Có Vạch Chia", "Băng Quấn Cổ Chân Bảo Vệ Khớp", "Dây Nhảy Thể Lực Lõi Thép Chống Rối", "Túi Trống Thể Thao Tập Gym Du Lịch"]
    }
    
    for i in range(1, 1201):
        cat = random.choice(categories)
        base_name = random.choice(prod_names[cat])
        adj = random.choice(adjectives)
        brand = random.choice(["Shopee Mall", "Shop Yêu Thích", "Shop Yêu Thích+", "Shop Thường"])
        p_name = f"[{brand}] {base_name} {adj} Mã SP-{i:04d}"
        
        # Giá từ 25k đến 5.5tr
        if cat in ["Điện Thoại & Phụ Kiện", "Máy Tính & Laptop", "Thiết Bị Điện Gia Dụng"]:
            price = random.randint(150, 4500) * 1000
        elif cat in ["Thời Trang Nam/Nữ", "Mỹ Phẩm & Làm Đẹp"]:
            price = random.randint(59, 990) * 1000
        else:
            price = random.randint(25, 650) * 1000
            
        discount_rate = random.choice([0, 5, 10, 15, 20, 25, 30, 40, 50, 60])
        is_megasale = 1 if discount_rate >= 30 or random.random() < 0.25 else 0
        shipping_fee = random.choice([0, 15000, 22000, 30000, 45000])
        
        # Rating & lượt tương tác
        shop_rating = round(random.uniform(3.8, 5.0), 2)
        rating_star = round(random.uniform(3.5, 5.0), 2)
        view_count = int(np.random.exponential(scale=15000) + 500)
        favorite_count = int(view_count * random.uniform(0.02, 0.15))
        historical_sold = int(view_count * random.uniform(0.03, 0.25)) + random.randint(10, 500)
        
        # Doanh số tháng (Monthly sold) phụ thuộc vào rating, giảm giá, megasale và vị trí
        base_monthly = int(historical_sold * random.uniform(0.1, 0.45))
        if is_megasale:
            base_monthly = int(base_monthly * random.uniform(1.3, 2.2))
        if rating_star < 4.0:
            base_monthly = int(base_monthly * 0.4)
            
        monthly_sold = max(5, base_monthly)
        rating_count = int(historical_sold * random.uniform(0.3, 0.7)) + 5
        
        # Chỉ số vận hành chuỗi cung ứng
        ship_on_time_rate = round(random.uniform(85.0, 99.8), 1)
        chat_response_rate = round(random.uniform(70.0, 100.0), 1)
        avg_delivery_days = round(random.uniform(1.5, 6.5), 1)
        
        # Rủi ro hoàn hàng và đánh giá xấu
        return_rate = round(random.uniform(0.5, 8.5) if rating_star >= 4.5 else random.uniform(4.0, 18.5), 2)
        bad_review_rate = round(random.uniform(0.5, 5.0) if rating_star >= 4.5 else random.uniform(8.0, 35.0), 2)
        
        # Nhãn tiềm năng tăng trưởng (Growth Potential Label)
        growth_score = (monthly_sold / 500) * 0.35 + (rating_star / 5.0) * 0.25 + (view_count / 10000) * 0.20 + (ship_on_time_rate / 100) * 0.20
        if growth_score > 1.2 and rating_star >= 4.4:
            growth_potential = "Tiềm Năng Cao"
        elif growth_score > 0.6 and rating_star >= 3.9:
            growth_potential = "Trung Bình"
        else:
            growth_potential = "Rủi Ro / Kém"
            
        # Nhãn Rủi ro vận hành (Risk Level)
        if return_rate > 10.0 or bad_review_rate > 15.0 or ship_on_time_rate < 88.0:
            risk_level = "Cao"
        elif return_rate > 5.0 or bad_review_rate > 8.0:
            risk_level = "Trung Bình"
        else:
            risk_level = "Thấp"
            
        product_rows.append({
            "product_id": f"SP_{i:05d}",
            "product_name": p_name,
            "category": cat,
            "shop_type": brand,
            "shop_location": random.choice(locations),
            "price": price,
            "discount_rate": discount_rate,
            "is_megasale": is_megasale,
            "shipping_fee": shipping_fee,
            "view_count": view_count,
            "favorite_count": favorite_count,
            "historical_sold": historical_sold,
            "monthly_sold": monthly_sold,
            "rating_star": rating_star,
            "rating_count": rating_count,
            "shop_rating": shop_rating,
            "chat_response_rate": chat_response_rate,
            "ship_on_time_rate": ship_on_time_rate,
            "avg_delivery_days": avg_delivery_days,
            "return_rate": return_rate,
            "bad_review_rate": bad_review_rate,
            "growth_potential": growth_potential,
            "risk_level": risk_level
        })
        
    df_products = pd.DataFrame(product_rows)
    df_products.to_csv(output_path / "shopee_products.csv", index=False, encoding="utf-8-sig")
    
    # -------------------------------------------------------------------------
    # B. TẠO DỮ LIỆU ĐÁNH GIÁ & BÌNH LUẬN TRUYỀN THÔNG XÃ HỘI (3,500 reviews)
    # -------------------------------------------------------------------------
    pos_reviews_pool = [
        "sp dùng siêu ok nha mn, shop giao hàng nhanh chóng, đóng gói cẩn thận 5 sao ❤️",
        "hàng chuẩn auth 100%, chất lượng quá đỉnh, giá cả lại rẻ hơn bên ngoài nhiều",
        "đã nhận đc hàng, bọc bóng khí kỹ càng, shipper thân thiện nhiệt tình cute xỉu",
        "chất vải đẹp mịn mát, form áo chuẩn không cần chỉnh, sẽ ủng hộ shop nhìu lần nữa",
        "giao hàng hỏa tốc trong 2h, sản phẩm nguyên seal, bảo hành chính hãng uy tín",
        "mua trúng đợt mega sale giá hạt dẻ dã man, chất lượng vượt xa mong đợi 💯🔥",
        "máy chạy êm ru, nhỏ gọn tiện lợi, shop rep tin nhắn tư vấn rất nhiệt tình luôn ạ",
        "đóng gói siêu chắc chắn, hộp không hề bị móp méo, hàng đẹp xịn sò đáng đồng tiền",
        "hàng xịn mlem lắm nha, shop nhiệt tình hướng dẫn sử dụng chi tiết, 10 điểm",
        "rất ưng ý với sản phẩm này, màu sắc y hình chụp, freeship còn áp thêm voucher xịn",
        "tuyệt vời ông mặt trời, dùng 1 tuần rồi pin vẫn trâu, đồ tốt giá okla",
        "shop đóng gói có tâm ghê, tặng kèm cả sticker và thư cảm ơn cute, rate 5 sao",
        "hàng giao đúng mẫu đúng size, chất lượng tốt so với giá thành, vote ủng hộ shop",
        "săn sale 0đ mà hàng nhận về chất lượng không khác gì mua tại showroom, quá đỉnh",
        "sp chính hãng tem mác đầy đủ, quét mã qr ra ngay thông tin, cực kỳ yên tâm mua sắm"
    ]
    
    neu_reviews_pool = [
        "hàng nhận đc bình thường, tiền nào của nấy, tạm chấp nhận đc với mức giá này",
        "giao hàng hơi lâu 1 xíu, đóng gói đơn giản nhưng may là không bị vỡ",
        "sp giống ảnh khoảng 80%, chất liệu trung bình, dùng tạm thì ok",
        "shop giao đúng số lượng nhưng màu sắc hơi khác ảnh mẫu 1 tông",
        "đã nhận đủ hàng, chất lượng ổn, thời gian giao hàng mất 4 ngày",
        "sản phẩm tạm ổn, mùi hơi nồng lúc mới bóc hộp, để vài hôm thì hết",
        "dùng cũng tạm được, không quá xuất sắc nhưng cũng không đến nỗi tệ",
        "đóng gói hộp hơi móp nhẹ, bên trong sản phẩm không ảnh hưởng gì",
        "shop rep ib hơi chậm nhưng nói chuyện lịch sự, hàng vừa vặn",
        "mua thử xem thế nào, thấy bình thường như các loại khác trên thị trường"
    ]
    
    neg_reviews_pool = [
        "quá thất vọng, hàng fake kém chất lượng, đặt màu đen giao màu trắng 😡",
        "shop lừa đảo, hàng dỏm dùng 2 ngày đã hỏng, nhắn tin khiếu nại thì bị chặn ko rep",
        "giao hàng siêu chậm, mất gần 10 ngày mới nhận được, shipper thái độ rất khó chịu",
        "đóng gói sơ sài cẩu thả, hộp bị móp méo vỡ nát hết đồ bên trong 😭👎",
        "chất vải mỏng dính như giấy, may ẩu chỉ thừa tùm lum, phí tiền mua",
        "hàng nhái rep 1:1 dở ẹc, ko giống như quảng cáo trên live tiktok xíu nào",
        "shop làm ăn vô trách nhiệm, giao thiếu hàng mà không chịu giải quyết hoàn tiền",
        "sp lỗi không khởi động được, yêu cầu đổi trả thì shop bảo do khách làm hỏng 🤬",
        "quá tệ hại, giá đắt mà nhận về món đồ như đồng nát, mn né shop này ra gấp",
        "scam trắng trợn, hình đăng một đằng gửi hàng một nẻo, 1 sao cũng không xứng đáng",
        "bùng đơn giao trễ làm lỡ việc của mình, cskh trả lời cụt lủn thiếu tôn trọng",
        "hàng kém chất lượng mùi nhựa hắc nồng nặc, không an toàn sử dụng, tẩy chay"
    ]
    
    channels = ["Shopee Review", "Facebook Group Review E-Commerce", "TikTok Shop Comment", "Voz/Tinhte Forum"]
    
    review_rows = []
    for r_idx in range(1, 3501):
        rand_p = random.choice(product_rows)
        # Tỷ lệ: 60% Tích cực, 20% Trung lập, 20% Tiêu cực
        sentiment_dice = random.random()
        if sentiment_dice < 0.60:
            raw_text = random.choice(pos_reviews_pool)
            sentiment_label = "Tích cực"
            rating = random.choice([5, 5, 5, 4])
        elif sentiment_dice < 0.80:
            raw_text = random.choice(neu_reviews_pool)
            sentiment_label = "Trung lập"
            rating = random.choice([3, 3, 4, 3])
        else:
            raw_text = random.choice(neg_reviews_pool)
            sentiment_label = "Tiêu cực"
            rating = random.choice([1, 1, 2, 1])
            
        # Thêm biến thể teencode ngẫu nhiên
        if random.random() < 0.4:
            raw_text = raw_text.replace("sản phẩm", "sp").replace("không", "k").replace("được", "đc").replace("cửa hàng", "shop")
            
        cleaned = clean_vietnamese_text(raw_text)
        score, calc_label, pos_words, neg_words = calculate_lexicon_sentiment_score(raw_text)
        
        review_rows.append({
            "review_id": f"REV_{r_idx:06d}",
            "product_id": rand_p["product_id"],
            "product_name": rand_p["product_name"],
            "category": rand_p["category"],
            "channel": random.choice(channels),
            "raw_comment": raw_text,
            "cleaned_comment": cleaned,
            "rating_star": rating,
            "sentiment_label": sentiment_label,
            "sentiment_score": score,
            "helpful_votes": random.randint(0, 45),
            "has_media": 1 if random.random() < 0.65 else 0
        })
        
    df_reviews = pd.DataFrame(review_rows)
    df_reviews.to_csv(output_path / "shopee_reviews.csv", index=False, encoding="utf-8-sig")
    
    # -------------------------------------------------------------------------
    # C. TẠO DỮ LIỆU GIAO DỊCH GIỎ HÀNG (MARKET BASKET CHO APRIORI)
    # -------------------------------------------------------------------------
    transaction_rows = []
    combos = [
        ["Củ Sạc Nhanh 65W", "Cáp Sạc Nhanh Type-C", "Kính Cường Lực Full Màn", "Ốp Lưng Chống Sốc"],
        ["Áo Thun Unisex Cotton", "Quần Jean Baggy", "Giày Sneaker Thể Thao"],
        ["Chuột Không Dây Silent", "Bàn Phím Cơ LED RGB", "Lót Chuột Gaming Khổ Lớn", "Giá Đỡ Laptop Nhôm"],
        ["Kem Chống Nắng Kiềm Dầu", "Nước Tẩy Trang Micellar", "Sữa Rửa Mặt Dịu Nhẹ", "Serum B5 Phục Hồi"],
        ["Nồi Chiên Không Dầu", "Bình Giữ Nhiệt Inox", "Ấm Siêu Tốc 2 Lớp"],
        ["Bình Sữa Chống Sặc", "Tã Bỉm Dán Siêu Thấm", "Khăn Ướt Kháng Khuẩn"]
    ]
    
    all_single_items = [item for sublist in combos for item in sublist] + [
        "Váy Nữ Dáng Xòe", "Đèn Bàn Học LED", "Máy Xay Sinh Tố Mini", "Thảm Tập Yoga", "Bình Nước 2L"
    ]
    
    for t_id in range(1, 1501):
        if random.random() < 0.7:
            chosen_combo = random.choice(combos)
            k = random.randint(2, len(chosen_combo))
            items = random.sample(chosen_combo, k)
            if random.random() < 0.3:
                items.append(random.choice(all_single_items))
        else:
            k = random.randint(2, 4)
            items = random.sample(all_single_items, k)
            
        unique_items = list(dict.fromkeys(items))
        transaction_rows.append({
            "transaction_id": f"TRX_{t_id:05d}",
            "items": ", ".join(unique_items),
            "item_count": len(unique_items),
            "payment_method": random.choice(["ShopeePay", "COD (Tiền mặt)", "Thẻ Tín Dụng/Ghi Nợ", "SPayLater (Mua trước trả sau)"])
        })
        
    df_transactions = pd.DataFrame(transaction_rows)
    df_transactions.to_csv(output_path / "shopee_transactions.csv", index=False, encoding="utf-8-sig")

    data_files = (
        "shopee_products.csv",
        "shopee_reviews.csv",
        "shopee_transactions.csv",
    )
    provenance = {
        "source_type": "synthetic_seeded_demo",
        "seed": 42,
        "generated_by": "data_processing.generate_shopee_datasets",
        "records": {
            "products": len(df_products),
            "reviews": len(df_reviews),
            "transactions": len(df_transactions),
        },
        "sha256": {
            name: hashlib.sha256((output_path / name).read_bytes()).hexdigest()
            for name in data_files
        },
    }
    with (output_path / "provenance.json").open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, ensure_ascii=False, indent=2)
    
    print(f"-> Đã tạo thành công bộ dữ liệu mô phỏng tại '{output_path}':")
    print(f"   + {len(df_products)} sản phẩm ({output_path / 'shopee_products.csv'})")
    print(f"   + {len(df_reviews)} đánh giá & MXH ({output_path / 'shopee_reviews.csv'})")
    print(f"   + {len(df_transactions)} giao dịch giỏ hàng ({output_path / 'shopee_transactions.csv'})")
    
    return {
        "products": df_products,
        "reviews": df_reviews,
        "transactions": df_transactions
    }

if __name__ == "__main__":
    generate_shopee_datasets()

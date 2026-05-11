import os
import time
from bing_image_downloader import downloader

def crawl_vietnamese_food_local(base_output_dir="D:\Learn Python\Vietnamese_Food_Project\data", samples_per_class=1100):

    # Tạo thư mục gốc nếu chưa có
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
        print(f"Đã tạo thư mục gốc tại: {base_output_dir}")

    food_classes = {
        'Banh_bot_loc': ['Bánh bột lọc Huế', 'Bánh bột lọc trần tôm thịt', 'Vietnamese clear shrimp dumplings'],
        'Banh_mi_Viet_Nam': ['Bánh mì Việt Nam', 'Bánh mì thịt chả', 'Vietnamese banh mi sandwich'],
        'Banh_xeo': ['Bánh xèo miền Tây', 'Bánh xèo tôm thịt giòn', 'Vietnamese sizzling pancake banh xeo'],
        'Bun_bo_Hue': ['Bún bò Huế ngon', 'Tô bún bò huế đầy đủ', 'Bun bo hue spicy beef noodle soup'],
        'Bun_thit_nuong': ['Bún thịt nướng chả giò', 'Tô bún thịt nướng mỡ hành', 'Vietnamese grilled pork noodles bun thit nuong'],
        'Cao_lau': ['Cao lầu Hội An', 'Tô cao lầu xá xíu', 'Cao lau noodles Hoi An'],
        'Com_tam': ['Cơm tấm sườn bì chả', 'Đĩa cơm tấm mỡ hành', 'Vietnamese broken rice com tam'],
        'Goi_cuon': ['Gỏi cuốn tôm thịt', 'Gỏi cuốn chấm tương đậu', 'Vietnamese fresh spring rolls goi cuon'],
        'Mi_Quang': ['Mì Quảng tôm thịt gà', 'Tô mì quảng đặc biệt', 'Mi quang turmeric noodles'],
        'Pho_bo': ['Phở bò Hà Nội', 'Tô phở bò tái nạm', 'Vietnamese beef pho noodle soup']
    }

    print(f"--- BẮT ĐẦU CRAWL DỮ LIỆU VỀ MÁY TÍNH ---")

    for class_name, keywords in food_classes.items():
        # Chia đều chỉ tiêu cho các từ khóa
        limit_per_keyword = samples_per_class // len(keywords)

        print(f"\nĐang xử lý món: {class_name}")
        for kw in keywords:
            try:
                downloader.download(
                    kw,
                    limit=limit_per_keyword,
                    output_dir=base_output_dir, # Lưu vào thư mục đã định nghĩa
                    adult_filter_off=True,
                    force_replace=False,
                    timeout=30,
                    verbose=False
                )
                time.sleep(1) # Nghỉ 1 giây để tránh bị chặn IP
            except Exception as e:
                print(f"       [Lỗi] Từ khóa '{kw}': {e}")

    print(f"\nHOÀN TẤT! Dữ liệu đã được lưu tại: {base_output_dir}")

if __name__ == "__main__":
    # Thay đổi đường dẫn này theo ý muốn của bạn
    MY_PATH = "D:/Dataset_Food_Project"
    crawl_vietnamese_food_local(base_output_dir=MY_PATH)
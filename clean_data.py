import os
from PIL import Image


def compute_image_hash(image_path):
    """
    Tính toán mã băm (Average Hash) để xác định các hình ảnh giống nhau.
    """
    try:
        with Image.open(image_path) as img:
            # Chuyển về thang độ xám và thu nhỏ về 8x8
            img_gray = img.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
            pixels = list(img_gray.getdata())
            avg = sum(pixels) / len(pixels)
            # Tạo chuỗi bit: 1 nếu pixel > trung bình, 0 nếu ngược lại
            bits = "".join(['1' if p > avg else '0' for p in pixels])
            return hex(int(bits, 2))[2:].zfill(16)
    except Exception:
        return None


def comprehensive_data_cleaning(dataset_dir):
    """
    Quy trình làm sạch dữ liệu:
    1. Xóa file lỗi/hỏng.
    2. Chuẩn hóa hệ màu về RGB.
    3. Xóa ảnh trùng lặp dựa trên mã băm.
    """
    if not os.path.exists(dataset_dir):
        print(f"Thư mục '{dataset_dir}' không tồn tại. Hãy kiểm tra lại đường dẫn!")
        return

    removed_corrupted = 0
    removed_duplicates = 0
    seen_hashes = set()

    # Lấy danh sách các thư mục con (các lớp món ăn)
    classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

    print("BẮT ĐẦU QUÁ TRÌNH LÀM SẠCH DỮ LIỆU...")

    for cls in classes:
        cls_dir = os.path.join(dataset_dir, cls)
        files = os.listdir(cls_dir)
        print(f" + Đang quét lớp: [{cls}] - Tổng số: {len(files)} tệp")

        for file_name in files:
            file_path = os.path.join(cls_dir, file_name)

            # 1. Kiểm tra tính toàn vẹn và chuẩn hóa màu
            try:
                # Kiểm tra xem file có thực sự là ảnh và không bị hỏng
                with Image.open(file_path) as img:
                    img.verify()

                    # Mở lại để xử lý (verify xong phải mở lại mới xử lý được)
                with Image.open(file_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                        img.save(file_path)
            except Exception:
                os.remove(file_path)
                removed_corrupted += 1
                continue

            # 2. Kiểm tra trùng lặp
            img_hash = compute_image_hash(file_path)
            if img_hash is None:
                if os.path.exists(file_path):
                    os.remove(file_path)
                removed_corrupted += 1
                continue

            if img_hash in seen_hashes:
                os.remove(file_path)
                removed_duplicates += 1
            else:
                seen_hashes.add(img_hash)

    print("\n" + "=" * 30)
    print("BÁO CÁO KẾT QUẢ LÀM SẠCH:")
    print(f" - Đã xóa file hỏng: {removed_corrupted:,} tệp")
    print(f" - Đã xóa file trùng: {removed_duplicates:,} tệp")
    print(f" - Dữ liệu sạch còn lại: {len(seen_hashes):,} hình ảnh")
    print("=" * 30)


if __name__ == "__main__":

    PATH_TO_DATA = r'D:\Learn Python\Vietnamese_Food_Project\data'

    comprehensive_data_cleaning(PATH_TO_DATA)
import os


def rename_images_in_folder(folder_path):
    # 1. Lấy danh sách tất cả các file trong thư mục và sắp xếp chúng
    try:
        files = os.listdir(folder_path)
        files.sort()  # Sắp xếp theo thứ tự bảng chữ cái để đảm bảo tính nhất quán
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy thư mục '{folder_path}'")
        return

    # 2. Khởi tạo biến đếm
    count = 1

    # 3. Định nghĩa các định dạng ảnh hợp lệ (tránh đổi tên nhầm file hệ thống hoặc file text)
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')

    print("Bắt đầu đổi tên file...")

    # 4. Duyệt qua từng file
    for filename in files:
        # Tách tên file và phần đuôi (extension)
        name, ext = os.path.splitext(filename)

        # Đưa đuôi file về chữ thường để kiểm tra cho chuẩn (VD: .JPG -> .jpg)
        ext_lower = ext.lower()

        # 5. Nếu file đó đúng là ảnh
        if ext_lower in valid_extensions:
            # Tạo đường dẫn đầy đủ cho file hiện tại
            old_filepath = os.path.join(folder_path, filename)

            # Tạo tên mới và đường dẫn mới (VD: 1.jpg, 2.png)
            new_filename = f"{count}{ext_lower}"
            new_filepath = os.path.join(folder_path, new_filename)

            # Đổi tên file
            try:
                os.rename(old_filepath, new_filepath)
                print(f"Đã đổi: {filename}  -->  {new_filename}")
                count += 1
            except Exception as e:
                print(f"Không thể đổi tên file {filename}. Lỗi: {e}")

    print(f"\nHoàn tất! Đã đổi tên thành công {count - 1} file ảnh.")


# ==========================================
# CÁCH SỬ DỤNG:
# Bạn chỉ cần thay đổi biến 'duong_dan_thu_muc' thành đường dẫn chứa ảnh của bạn.
# Lưu ý: Thêm chữ 'r' phía trước chuỗi đường dẫn trên Windows để tránh lỗi ký tự đặc biệt.
# ==========================================

duong_dan_thu_muc = r"data\Phở_bò"
rename_images_in_folder(duong_dan_thu_muc)
import os
import shutil
import random

# Đường dẫn thư mục chứa ảnh đã crawl và lọc
root_dir = 'data'
classes = os.listdir(root_dir)

# Tạo cấu trúc thư mục mới
base_dir = 'dataset_final'
for split in ['train', 'val', 'test']:
    for cls in classes:
        os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)

# Chia dữ liệu
for cls in classes:
    src_path = os.path.join(root_dir, cls)
    images = os.listdir(src_path)
    random.shuffle(images)

    # Tính toán số lượng
    train_count = int(len(images) * 0.8)
    val_count = int(len(images) * 0.1)

    train_imgs = images[:train_count]
    val_imgs = images[train_count:train_count + val_count]
    test_imgs = images[train_count + val_count:]

    # Copy file vào các thư mục tương ứng
    for img in train_imgs:
        shutil.copy(os.path.join(src_path, img), os.path.join(base_dir, 'train', cls, img))
    for img in val_imgs:
        shutil.copy(os.path.join(src_path, img), os.path.join(base_dir, 'val', cls, img))
    for img in test_imgs:
        shutil.copy(os.path.join(src_path, img), os.path.join(base_dir, 'test', cls, img))

print("--- Đã chia dữ liệu thành công! ---")
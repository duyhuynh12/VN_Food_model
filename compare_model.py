import os
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import models


# =====================================================================
# PHẦN 1: ĐỊNH NGHĨA KIẾN TRÚC 2 MÔ HÌNH TOÀN VẸN
# =====================================================================

# --- KIẾN TRÚC 1: MobileNetV2 + DeepSE (Từ file Notebook KHDL_v2 của bạn) ---
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class InvertedResidualSE(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidualSE, self).__init__()
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True),
            SEBlock(hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res_connect else self.conv(x)


class MobileNetV2_DeepSE(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetV2_DeepSE, self).__init__()
        self.configs = [
            [1, 16, 1, 1], [6, 24, 3, 2], [6, 32, 4, 2], [6, 64, 6, 2],
            [6, 96, 4, 1], [6, 160, 4, 2], [6, 320, 1, 1],
        ]
        input_channel, last_channel = 32, 1280
        features = [nn.Conv2d(3, input_channel, 3, 2, 1, bias=False), nn.BatchNorm2d(input_channel),
                    nn.ReLU6(inplace=True)]
        for t, c, n, s in self.configs:
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidualSE(input_channel, c, stride, expand_ratio=t))
                input_channel = c
        features.append(nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False))
        features.append(nn.BatchNorm2d(last_channel))
        features.append(nn.ReLU6(inplace=True))
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(last_channel, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        return self.classifier(x)


# --- KIẾN TRÚC 2: ResNet50 Tiêu chuẩn ---
def create_resnet50(num_classes=10):
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


# =====================================================================
# PHẦN 2: HÀM ĐÁNH GIÁ VÀ ĐO ĐẠC HIỆU NĂNG ĐỒNG THỜI
# =====================================================================
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    # Đo đạc tốc độ suy luận
    start_time = time.time()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    end_time = time.time()
    total_time = (end_time - start_time) * 1000  # Đổi sang mili-giây (ms)
    avg_latency = total_time / len(dataloader.dataset)  # Thời gian trên mỗi ảnh

    return np.array(all_labels), np.array(all_preds), avg_latency


# =====================================================================
# PHẦN 3: ĐƯỜNG ỐNG XỬ LÝ CHÍNH
# =====================================================================
def run_comparison(test_dir, weights_mobilenet, weights_resnet50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Đang chạy đánh giá trên thiết bị: {device}")

    # Bộ tiền xử lý chuẩn hóa đồng bộ với quá trình huấn luyện
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    class_names = test_dataset.classes
    num_classes = len(class_names)

    # --- 1. Load mô hình MobileNetV2-DeepSE ---
    model_mono = MobileNetV2_DeepSE(num_classes=num_classes).to(device)
    model_mono.load_state_dict(torch.load(weights_mobilenet, map_location=device))

    # --- 2. Load mô hình ResNet50 ---
    model_res = create_resnet50(num_classes=num_classes).to(device)
    model_res.load_state_dict(torch.load(weights_resnet50, map_location=device))

    # --- 3. Tiến hành đánh giá ---
    print("⏳ Đang kiểm tra mô hình MobileNetV2_DeepSE...")
    labels_mono, preds_mono, speed_mono = evaluate_model(model_mono, test_loader, device)

    print("⏳ Đang kiểm tra mô hình ResNet50...")
    labels_res, preds_res, speed_res = evaluate_model(model_res, test_loader, device)

    # --- 4. Tính toán độ chính xác chi tiết theo từng lớp ---
    acc_per_class_mono = []
    acc_per_class_res = []

    for i in range(num_classes):
        idx = (labels_mono == i)
        acc_mono = np.sum(preds_mono[idx] == labels_mono[idx]) / np.sum(idx) * 100
        acc_res = np.sum(preds_res[idx] == labels_res[idx]) / np.sum(idx) * 100
        acc_per_class_mono.append(acc_mono)
        acc_per_class_res.append(acc_res)

    # --- 5. Lấy dung lượng file mô hình (MB) ---
    size_mono = os.path.getsize(weights_mobilenet) / (1024 * 1024)
    size_res = os.path.getsize(weights_resnet50) / (1024 * 1024)

    # =====================================================================
    # PHẦN 4: VẼ BIỂU ĐỒ SO SÁNH ĐA TIÊU CHÍ
    # =====================================================================
    print("📊 Đang sinh đồ thị so sánh...")

    # ĐỒ THỊ 1: SO SÁNH ĐỘ CHÍNH XÁC TỪNG LỚP MÓN ĂN (GROUPED BAR CHART)
    x = np.arange(len(class_names))
    width = 0.35  # Độ rộng của mỗi cột dơn

    fig, ax = plt.subplots(figsize=(14, 7))
    rects1 = ax.bar(x - width / 2, acc_per_class_mono, width, label='MobileNetV2-DeepSE', color='#4CAF50',
                    edgecolor='black')
    rects2 = ax.bar(x + width / 2, acc_per_class_res, width, label='ResNet50', color='#2196F3', edgecolor='black')

    ax.set_ylabel('Độ chính xác (%)', fontsize=12, fontweight='bold')
    ax.set_title('BIỂU ĐỒ SO SÁNH ĐỘ CHÍNH XÁC CHI TIẾT THEO TỪNG LỚP MÓN ĂN', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha='right', fontsize=11)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle=':', alpha=0.6)

    # Gán số liệu % lên đầu các cột ảnh
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # Lệch trục đứng lên 3pt
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    plt.savefig('comparison_accuracy_per_class.png', dpi=300)
    plt.show()

    # ĐỒ THỊ 2: SO SÁNH HIỆU NĂNG PHẦN CỨNG (DUNG LƯỢNG & TỐC ĐỘ)
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    models_label = ['MobileNetV2-DeepSE', 'ResNet50']

    # Vẽ biểu đồ so sánh dung lượng mạng (Càng nhỏ càng tốt)
    ax1.bar(models_label, [size_mono, size_res], color=['#4CAF50', '#2196F3'], edgecolor='black', width=0.5)
    ax1.set_ylabel('Dung lượng tệp trọng số (MB)', fontsize=12, fontweight='bold')
    ax1.set_title('SO SÁNH DUNG LƯỢNG MÔ HÌNH\n(Kích thước file càng nhỏ càng tối ưu)', fontsize=12, fontweight='bold')
    for i, v in enumerate([size_mono, size_res]):
        ax1.text(i, v + (max([size_mono, size_res]) * 0.02), f"{v:.2f} MB", ha='center', fontweight='bold')

    # Vẽ biểu đồ tốc độ phản hồi/độ trễ (Càng nhỏ càng nhanh)
    ax2.bar(models_label, [speed_mono, speed_res], color=['#4CAF50', '#2196F3'], edgecolor='black', width=0.5)
    ax2.set_ylabel('Thời gian xử lý / 1 hình ảnh (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('SO SÁNH ĐỘ TRỄ SUY LUẬN (LATENCY)\n(Thời gian càng thấp tốc độ càng nhanh)', fontsize=12,
                  fontweight='bold')
    for i, v in enumerate([speed_mono, speed_res]):
        ax2.text(i, v + (max([speed_mono, speed_res]) * 0.02), f"{v:.2f} ms", ha='center', fontweight='bold')

    fig2.tight_layout()
    # plt.savefig('comparison_hardware_performance.png', dpi=300)
    # plt.show()
    #
    # print("\n✅ Đã xuất thành công 2 file đồ thị chuẩn báo cáo:")
    # print(" 1. 'comparison_accuracy_per_class.png' (Độ chính xác từng lớp)")
    # print(" 2. 'comparison_hardware_performance.png' (Dung lượng và Độ trễ mạng)")


# ==========================================
# KHỞI CHẠY CHƯƠNG TRÌNH
# ==========================================
if __name__ == "__main__":
    # 🛑 SỬA ĐƯỜNG DẪN THỰC TẾ TRÊN MÁY TÍNH CỦA BẠN VÀO ĐÂY:
    PATH_TEST_DATA = r'D:\Learn Python\Vietnamese_Food_Project\dataset_final\test'
    FILE_MOBILENET = r'D:\Learn Python\Vietnamese_Food_Project\best_deep_se_model.pth'
    FILE_RESNET50 = r'D:\Learn Python\Vietnamese_Food_Project\best_resnet50_food.pth'

    run_comparison(PATH_TEST_DATA, FILE_MOBILENET, FILE_RESNET50)
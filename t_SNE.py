import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# ==========================================
# PHẦN 1: ĐỊNH NGHĨA KIẾN TRÚC MÔ HÌNH (GIỮ NGUYÊN)
# ==========================================
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


class FeatureExtractor_MobileNetV2_DeepSE(nn.Module):
    def __init__(self):
        super(FeatureExtractor_MobileNetV2_DeepSE, self).__init__()
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

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        return x


# ==========================================
# PHẦN 2: HÀM TRÍCH XUẤT VÀ VẼ ĐỒ THỊ
# ==========================================
def extract_and_visualize_features(dataset_dir, weights_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang chạy trên: {device}")

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(dataset_dir, transform=eval_transform)
    # Lấy tối đa 1000 mẫu để vẽ t-SNE (quá nhiều sẽ bị rối và chạy rất lâu)
    num_samples = min(1000, len(dataset))
    subset_indices = np.random.choice(len(dataset), num_samples, replace=False)
    subset = torch.utils.data.Subset(dataset, subset_indices)
    dataloader = DataLoader(subset, batch_size=32, shuffle=False)

    model = FeatureExtractor_MobileNetV2_DeepSE().to(device)

    if weights_path and os.path.exists(weights_path):
        try:
            # map_location=device giúp load model lên CPU nếu máy không có card đồ họa
            full_state_dict = torch.load(weights_path, map_location=device)
            # Lọc bỏ các trọng số của lớp classifier (chỉ lấy phần backbone)
            feature_state = {k: v for k, v in full_state_dict.items() if k.startswith("features.")}
            model.load_state_dict(feature_state)
            print("Đã nạp thành công trọng số đã huấn luyện!")
        except Exception as e:
            print(f"Cảnh báo: Không nạp được trọng số ({e}). Đang dùng trọng số ngẫu nhiên.")

    model.eval()
    features_list, labels_list = [], []

    print("🔍 Đang trích xuất đặc trưng (Feature Extraction)...")
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            feats = model(inputs)
            features_list.append(feats.cpu().numpy())
            labels_list.append(targets.numpy())

    X_features = np.vstack(features_list)
    y_labels = np.concatenate(labels_list)
    class_names = dataset.classes

    print(f"Đang giảm chiều dữ liệu (PCA -> t-SNE)...")
    # Bước 1: PCA giảm xuống 50 chiều để khử nhiễu
    X_pca = PCA(n_components=50, random_state=42).fit_transform(X_features)
    # Bước 2: t-SNE đưa về 2D để vẽ đồ thị
    X_tsne = TSNE(n_components=2, perplexity=30, random_state=42,init='pca', learning_rate='auto').fit_transform(X_pca)

    # VẼ ĐỒ THỊ
    plt.figure(figsize=(12, 8))
    # Sử dụng bảng màu đa dạng
    cmap = plt.get_cmap('tab10')

    for i, cls_name in enumerate(class_names):
        idx = (y_labels == i)
        plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=cls_name, alpha=0.7, s=40)

    plt.title("TRỰC QUAN HÓA ĐẶC TRƯNG MÓN ĂN (t-SNE VISUALIZATION)", fontsize=15, fontweight='bold')
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Danh sách món")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()


    plt.savefig('food_tsne_plot.png', dpi=300)
    print("Hoàn tất! Ảnh đồ thị đã được lưu tại 'food_tsne_plot.png'")
    plt.show()


# ==========================================
# CHẠY CHƯƠNG TRÌNH
# ==========================================
if __name__ == "__main__":

    dataset_path = r'D:\Learn Python\Vietnamese_Food_Project\data'
    weights_path = r'best_deep_se_model.pth'

    extract_and_visualize_features(dataset_path, weights_path)
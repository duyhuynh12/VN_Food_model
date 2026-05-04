import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from torchvision import transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import numpy as np

# Ngưỡng tin cậy để xác định "Món ăn chưa được học"
CONFIDENCE_THRESHOLD = 0.60


# ==========================================
# PHẦN 1: ĐỊNH NGHĨA LẠI KIẾN TRÚC MÔ HÌNH
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
            layers.append(nn.BatchNorm2d(hidden_dim));
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
        features.append(nn.BatchNorm2d(last_channel));
        features.append(nn.ReLU6(inplace=True))
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(last_channel, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        return self.classifier(x)


# ==========================================
# PHẦN 2: THIẾT LẬP VÀ NẠP TRỌNG SỐ MÔ HÌNH
# ==========================================
MODEL_PATH = 'best_deep_se_model.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = [
    'Bánh bột lọc', 'Bánh mì Việt Nam', 'Bánh xèo', 'Bún bò Huế',
    'Bún thịt nướng', 'Cao lầu', 'Cơm tấm', 'Gỏi cuốn', 'Mì Quảng', 'Phở bò'
]

model = MobileNetV2_DeepSE(num_classes=len(class_names)).to(device)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Đã nạp mô hình thành công!")
else:
    print(f"Lỗi: Không tìm thấy file model tại '{MODEL_PATH}'. Hãy kiểm tra lại!")
    exit()


# ==========================================
# PHẦN 3: XÂY DỰNG GIAO DIỆN ỨNG DỤNG (APP)
# ==========================================
class FoodRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Nhận Diện Món Ăn Việt Nam")
        self.root.geometry("1000x550")

        # 1. Khung chứa nút bấm (Luôn hiện ở trên cùng)
        self.top_frame = tk.Frame(self.root, pady=10)
        self.top_frame.pack(side=tk.TOP, fill=tk.X)

        self.btn_select = tk.Button(
            self.top_frame,
            text="CHỌN ẢNH ĐỂ NHẬN DIỆN",
            command=self.select_and_predict,
            font=("Arial", 14, "bold"),
            bg="#4CAF50",
            fg="white",
            padx=20, pady=5
        )
        self.btn_select.pack()

        # 2. Khung chứa biểu đồ Matplotlib (Nhúng vào Tkinter)
        # Tạo sẵn 1 figure trống với 2 cột (1 cho ảnh, 1 cho biểu đồ)
        self.fig, self.axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Tiền xử lý
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Hiển thị hướng dẫn lúc mới mở app
        self.axes[0].text(0.5, 0.5, "Chưa có ảnh nào được chọn.\nHãy bấm nút bên trên!",
                          ha='center', va='center', fontsize=12)
        self.axes[0].axis('off')
        self.axes[1].axis('off')
        self.canvas.draw()

    def select_and_predict(self):
        # Mở hộp thoại chọn 1 file ảnh
        img_path = filedialog.askopenfilename(
            title='Chọn 1 ảnh món ăn',
            filetypes=[('Image Files', '*.jpg *.jpeg *.png *.webp')]
        )

        if not img_path:
            return  # Nếu ấn Cancel thì không làm gì cả

        # Đọc ảnh và dự đoán
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            prob, preds = torch.max(probabilities, 0)

        prob_value = prob.item()
        predicted_class = class_names[preds.item()]
        probs_np = probabilities.cpu().numpy() * 100

        # Xóa nội dung của lần hiển thị trước
        self.axes[0].clear()
        self.axes[1].clear()

        # --- KIỂM TRA NGƯỠNG TIN CẬY ---
        if prob_value < CONFIDENCE_THRESHOLD:
            title_text = f"MÓN ĂN CHƯA ĐƯỢC HỌC!\n(Giống '{predicted_class}' nhất: {prob_value * 100:.2f}%)"
            title_color = 'red'
        else:
            title_text = f"Dự đoán: {predicted_class}\nĐộ tin cậy: {prob_value * 100:.2f}%"
            title_color = 'green'

        # --- VẼ ẢNH (BÊN TRÁI) ---
        self.axes[0].imshow(img)
        self.axes[0].set_title(title_text, color=title_color, fontweight='bold', fontsize=14)
        self.axes[0].axis('off')

        # --- VẼ BIỂU ĐỒ (BÊN PHẢI) ---
        y_pos = np.arange(len(class_names))
        colors = ['skyblue'] * len(class_names)
        colors[preds.item()] = 'darkorange'  # Tô màu cam cho cột dự đoán cao nhất

        bars = self.axes[1].barh(y_pos, probs_np, color=colors)
        self.axes[1].set_yticks(y_pos)
        self.axes[1].set_yticklabels(class_names, fontsize=10)
        self.axes[1].invert_yaxis()  # Đọc từ trên xuống
        self.axes[1].set_xlabel('Tỉ lệ tin cậy (%)', fontsize=12)
        self.axes[1].set_xlim(0, 100)
        self.axes[1].set_title('Phân phối dự đoán', fontsize=12)

        # Hiện số % lên từng thanh
        for bar in bars:
            width = bar.get_width()
            if width > 1:
                self.axes[1].text(width + 1, bar.get_y() + bar.get_height() / 2,
                                  f'{width:.1f}%', va='center')

        self.fig.tight_layout()

        # Lệnh quan trọng: Cập nhật lại giao diện canvas sau khi vẽ xong
        self.canvas.draw()


# ==========================================
# KHỞI CHẠY ỨNG DỤNG
# ==========================================
if __name__ == '__main__':
    # Tạo cửa sổ giao diện chính
    root = tk.Tk()
    # Khởi tạo App
    app = FoodRecognitionApp(root)
    # Vòng lặp giữ cho cửa sổ luôn mở
    root.mainloop()
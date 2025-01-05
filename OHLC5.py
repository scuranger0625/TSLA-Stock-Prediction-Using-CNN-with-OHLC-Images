import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import cv2
from datetime import timedelta, datetime

# === 1. 取得 TSLA 資料並處理 ===
tsla_data = yf.download("TSLA", start="2010-01-01", end="2024-12-01")
tsla_data.reset_index(inplace=True)

# 標準化數據
scaler = MinMaxScaler()
tsla_data[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(
    tsla_data[['Open', 'High', 'Low', 'Close', 'Volume']]
)

# === 2. 將市場數據轉換為黑白圖像 ===
def generate_OHLC_image(data, window_size=10, img_size=(64, 64)):
    image = np.zeros(img_size, dtype=np.uint8)
    high_low_range = data['High'].max() - data['Low'].min()
    if high_low_range == 0:
        return None  # 避免無效的數據

    scaled_data = ((data[['Open', 'High', 'Low', 'Close']] - data['Low'].min()) / high_low_range * (img_size[0]-1)).astype(int)
    for idx, row in scaled_data.iterrows():
        x = (idx - data.index[0]) * (img_size[1] // window_size)
        cv2.line(image, (x, img_size[0]-row['High']), (x, img_size[0]-row['Low']), 255, 1)
        cv2.line(image, (x, img_size[0]-row['Open']), (x+2, img_size[0]-row['Open']), 255, 1)
        cv2.line(image, (x, img_size[0]-row['Close']), (x+2, img_size[0]-row['Close']), 255, 1)
    return torch.tensor(image[np.newaxis, :, :], dtype=torch.float32)

# 自定義 Dataset 類
class StockImageDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# === 3. 定義 CNN 模型 ===
class CNN_5_model(nn.Module):
    def __init__(self):
        super(CNN_5_model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# === 4. 訓練模型並保存 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_5_model().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# 真實數據
X_train = []
y_train = []

# 生成數據並過濾無效圖像
for i in range(len(tsla_data) - 10):
    image = generate_OHLC_image(tsla_data.iloc[i:i+10])
    if image is not None:  # 確保圖像有效
        X_train.append(image)
        # 標籤：下一天的Close價格是否高於當前視窗的最後一天Close價格
        label = 1 if tsla_data['Close'].iloc[i+10] > tsla_data['Close'].iloc[i+9] else 0
        y_train.append(label)

# 將數據轉換為Tensor
X_train = torch.stack(X_train)  # 堆疊為Tensor
y_train = torch.tensor(y_train, dtype=torch.float32)  # 標籤轉為Tensor

# 檢查數據長度是否匹配
assert len(X_train) == len(y_train), "X_train和y_train長度不一致！"

# 創建數據集和DataLoader
train_loader = DataLoader(StockImageDataset(X_train, y_train), batch_size=32, shuffle=True)


# 訓練
model.train()
for epoch in range(10):
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# 保存模型
torch.save(model.state_dict(), "best_model.pth")
print("模型已保存至 'best_model.pth'")

# === 5. 使用者輸入日期並測試 ===
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# 輸入說明
print("請輸入一個有效的日期 (格式: YYYY-MM-DD)，例如: 2022-12-15")
input_date = input("請輸入要預測的日期: ")
input_date = datetime.strptime(input_date, "%Y-%m-%d")

# 檢查數據
idx = tsla_data.index[tsla_data['Date'] == input_date].tolist()
if not idx or idx[0] < 10:
    print("無法找到足夠數據或指定日期不存在，請重新輸入。")
else:
    idx = idx[0]
    input_window = tsla_data.iloc[idx-10:idx]
    test_image = generate_OHLC_image(input_window)
    if test_image is None:
        print("無法生成圖像，請確認輸入的日期。")
    else:
        test_image = test_image.to(device)
        prediction = model(test_image.unsqueeze(0)).item()
        print(f"{input_date.date()} 的預測結果: {'上漲' if prediction > 0.5 else '下跌'} (機率: {prediction:.4f})")

        # 可視化
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(test_image.squeeze(0).cpu(), cmap='gray')
        plt.title(f"Grayscale OHLC Image\nDate: {input_date.date()}")
        plt.axis('off')

        # 修改時間範圍，僅顯示過去10天的數據
        start_date = input_date - timedelta(days=10)
        end_date = input_date - timedelta(days=1)  # 預測日的前一天

        # 獲取真實數據範圍
        actual_data = yf.download("TSLA", start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

        # 可視化盒鬚圖
        plt.subplot(1, 2, 2)
        plt.boxplot(actual_data['Close'], vert=True, patch_artist=True, boxprops=dict(facecolor="lightblue"))
        plt.title(f"Close Price Distribution\n{start_date.date()} ~ {end_date.date()}")
        plt.ylabel('Close Price')
        plt.xticks([1], ['Close Price'])

        plt.tight_layout()
        plt.show()

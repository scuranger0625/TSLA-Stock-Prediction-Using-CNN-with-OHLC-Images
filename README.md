# Read.me 文件模板

## **專案名稱**

請在此填寫專案的名稱，例如：**股票收益率預測與交易策略**。

---

## **專案描述**

簡要描述專案的目標與功能，例如：

> 本專案旨在利用深度學習技術，基於股票市場的 OHLC（開盤價、高價、低價、收盤價）數據，預測未來的收益率，並為投資者設計和優化交易策略。

---

## **技術架構與工具**

### **1. 使用的技術框架**
- **程式語言**：Python 3.12
- **深度學習框架**：PyTorch
- **數據處理與預處理**：
  - yFinance：股票數據下載
  - NumPy：數值計算
  - Scikit-learn：數據標準化
- **可視化工具**：
  - Matplotlib：圖表生成
  - OpenCV：OHLC 圖像生成

### **2. 模型架構**
- **卷積神經網絡（CNN）**：用於從 OHLC 圖像中提取特徵並進行收益率的迴歸預測。
- **優化算法**：Adam Optimizer
- **損失函數**：均方誤差（MSELoss）

---

## **資料說明**

### **1. 數據來源**
- 數據來自 Yahoo Finance（通過 `yfinance` 套件下載）。
- 包含的字段：
  - 開盤價（Open）
  - 收盤價（Close）
  - 最高價（High）
  - 最低價（Low）
  - 成交量（Volume）

### **2. 數據處理**
- 使用 MinMaxScaler 將 OHLC 數據標準化到 [0, 1] 範圍。
- 將 30 天的數據轉換為 64x64 像素的黑白圖像。

---

## **系統需求與環境**

### **1. 硬體需求**
- GPU 支援（推薦 CUDA 支援的顯卡）

### **2. 軟體需求**
- Python 版本：3.12
- 必需套件：
  ```bash
  pip install torch torchvision yfinance numpy scikit-learn matplotlib opencv-python
  ```

---

## **執行步驟**

### **1. 安裝環境**
安裝必要的 Python 套件：
```bash
pip install -r requirements.txt
```

### **2. 運行程式**
執行主程式：
```bash
python OHLC8.py
```
輸入提示：
- 輸入股票代號（例如 TSLA）。
- 輸入訓練次數（例如 50）。

### **3. 輸出結果**
- 訓練過程中的損失值（Loss）。
- 預測的未來收益率。
- 可視化的 OHLC 圖像與預測結果。

---

## **檔案結構**
```
project_directory/
|-- OHLC8.py                 # 主程式
|-- requirements.txt         # 必需的 Python 套件
|-- data/                    # 儲存下載的股票數據
|-- outputs/                 # 保存的模型權重與結果
|-- README.md                # 本文件
```

---

## **後續改進**
- 支援更多技術指標（如 RSI、MACD）。
- 優化 CNN 模型結構。
- 引入多目標預測（例如長期和短期收益率）。


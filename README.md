# TSLA Stock Prediction Using CNN with OHLC Images

## 概述

此專案旨在通過生成特斯拉 (TSLA) 的股市數據 OHLC (Open-High-Low-Close) 圖像，利用卷積神經網絡 (CNN) 模型進行股價上漲或下跌的預測。該專案運用深度學習技術將財務數據轉化為圖像並進行二分類。

## 功能
1. 從 Yahoo Finance 獲取 TSLA 股票數據，並進行標準化處理。
2. 將財務數據轉換為黑白 OHLC 圖像，用於模型訓練。
3. 使用 CNN 模型訓練和預測股票價格是否上漲。
4. 支援輸入指定日期，並輸出該日期的預測結果。
5. 提供圖像和數據的可視化功能。

## 系統需求

### 硬體需求
- CPU：一般現代處理器即可，但建議使用支援 CUDA 的 GPU 以加速訓練。
- RAM：至少 8GB。

### 軟體需求
- 作業系統：Windows、Linux 或 macOS。
- Python 版本：3.7 以上。
- 必須的 Python 套件：
  - `torch`
  - `numpy`
  - `pandas`
  - `yfinance`
  - `scikit-learn`
  - `matplotlib`
  - `opencv-python`

## 安裝與運行

### 安裝步驟
1. 安裝 Python：確保系統中已安裝 Python 3.7 以上版本。
2. 克隆或下載此專案到本地環境。
3. 使用以下指令安裝所需的 Python 套件：
   ```bash
   pip install torch numpy pandas yfinance scikit-learn matplotlib opencv-python
   ```
4. 確認安裝成功後，運行程式：
   ```bash
   python script_name.py
   ```

### 運行步驟
1. 啟動程式，程式會自動下載 TSLA 股票數據並訓練模型。
2. 模型訓練完成後，模型將被保存為 `best_model.pth`。
3. 程式運行時，輸入指定日期 (格式: YYYY-MM-DD)。例如：`2022-12-15`。
4. 程式將顯示該日期的預測結果及相關可視化圖像。

## 程式結構

### 文件與功能
- **OHLC 圖像生成 (generate_OHLC_image)**
  - 將股票數據轉換為 64x64 的黑白圖像，表示每段時間內的 OHLC 數據。

- **自定義 Dataset 類 (StockImageDataset)**
  - 將生成的圖像與標籤組合成 PyTorch 支援的數據集。

- **CNN 模型 (CNN_5_model)**
  - 三層卷積層，支援特徵提取及分類。

- **訓練模型**
  - 使用二元交叉熵損失函數 (BCELoss)，Adam 優化器進行模型訓練。

- **測試功能**
  - 根據輸入的日期生成測試圖像，並輸出預測結果與信心機率。

### 可視化
- **灰階 OHLC 圖像**：展示輸入數據轉換後的黑白圖像。
- **收盤價格盒鬚圖**：顯示過去 10 天內收盤價格的分布情況。

## 注意事項
1. **資料完整性檢查**
   - 確保輸入的日期有效且有足夠的歷史數據。
2. **無效數據處理**
   - 若 OHLC 圖像生成失敗，程式將提示用戶重新輸入。
3. **GPU 支援**
   - 如果系統支援 CUDA，程式會自動利用 GPU 加速模型訓練與推論。

## 範例輸出
- **輸入**: `2022-12-15`
- **輸出**:
  ```
  2022-12-15 的預測結果: 上漲 (機率: 0.7632)
  ```
- **圖像可視化**:
  - 左圖：64x64 灰階 OHLC 圖像。
  - 右圖：過去 10 天收盤價格的分布盒鬚圖。

## 授權
本專案使用 MIT 開源許可證。


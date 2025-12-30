import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import joblib  # Scaler'ı kaydetmek için
from sklearn.preprocessing import MinMaxScaler
from model import LSTMModelDemo

#VERİ SETİNİN HAZIRLANMASI---
symbol = "AAPL"
print(f"{symbol} verisi indiriliyor...")
df = yf.download(symbol, start="2022-01-01", end="2024-01-01")
data = df['Close'].values.reshape(-1, 1)

#Normalizasyon
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

X = []
y = []
seq_length = 4  # Son 4 güne bak -> 5. günü tahmin et

for i in range(len(data_scaled) - seq_length):
    X.append(data_scaled[i:i + seq_length])
    y.append(data_scaled[i + seq_length])

X = np.array(X)
y = np.array(y)

# Tensor'a çevirme
X = torch.tensor(X, dtype=torch.float32) # (Batch, Seq_Len, Feature)
y = torch.tensor(y, dtype=torch.float32)

#MODEL KURULUMU ---
model = LSTMModelDemo()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#EĞİTİM (TRAINING) ---
epochs = 200 
print("Eğitim başlıyor...")

loss_history = []

for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")


torch.save(model.state_dict(), "lstm_stock_model.pth")
joblib.dump(scaler, "scaler.save")
print("\nModel ve Scaler kaydedildi.")

#GÖRSELLEŞTİRME---
model.eval()
with torch.no_grad():
    predictions_scaled = model(X).numpy()

# Tahminleri ve gerçek verileri orijinal boyutlarına döndür
predictions = scaler.inverse_transform(predictions_scaled)
actuals = scaler.inverse_transform(y.numpy())

# Hataları hesapla (Residuals)
errors = actuals - predictions

#(4'lü Dashboard) ---
plt.figure(figsize=(16, 10))

# 1. GRAFİK: Zaman Serisi Tahmini
plt.subplot(2, 2, 1)
plt.plot(actuals, label='Gerçek Fiyat', color='blue', linewidth=1.5)
plt.plot(predictions, label='Model Tahmini', color='red', linestyle='--', linewidth=1.5)
plt.title(f'{symbol} Fiyat Tahmini (Zaman Serisi)')
plt.xlabel('Günler')
plt.ylabel('Fiyat ($)')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. GRAFİK: Loss (Kayıp) Eğrisi - Öğrenme Süreci
plt.subplot(2, 2, 2)
plt.plot(loss_history, label='Training Loss', color='purple')
plt.title('Model Eğitim Performansı (Loss Curve)')
plt.xlabel('Epoch Sayısı')
plt.ylabel('Hata (MSE)')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. GRAFİK: Scatter Plot (Korelasyon Analizi)
plt.subplot(2, 2, 3)
plt.scatter(actuals, predictions, color='green', alpha=0.5)
# İdeal tahmin doğrusu (y=x)
min_val = min(actuals.min(), predictions.min())
max_val = max(actuals.max(), predictions.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='İdeal Doğru')
plt.title('Gerçek vs Tahmin Korelasyonu')
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')
plt.legend()
plt.grid(True, alpha=0.3)

# 4. GRAFİK: Hata Dağılımı (Histogram)
plt.subplot(2, 2, 4)
plt.hist(errors, bins=30, color='orange', edgecolor='black', alpha=0.7)
plt.axvline(x=0, color='black', linestyle='--', linewidth=2)
plt.title('Hata Dağılımı (Residuals)')
plt.xlabel('Hata Miktarı ($)')
plt.ylabel('Frekans')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('detayli_analiz_dashboard.png', dpi=300)
print("\nDetaylı analiz grafiği 'detayli_analiz_dashboard.png' olarak kaydedildi.")
plt.show()

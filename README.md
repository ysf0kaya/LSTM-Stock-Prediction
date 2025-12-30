# 📈 Apple (AAPL) Stock Price Prediction using LSTM

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch)
![Gradio](https://img.shields.io/badge/Gradio-Interactive%20Demo-orange?style=for-the-badge)

Bu proje, **Derin Öğrenme (Deep Learning)** yöntemlerini kullanarak finansal zaman serisi tahmini yapmayı amaçlayan akademik bir çalışmadır. **Apple Inc. (AAPL)** hisse senetlerinin geçmiş fiyat hareketlerini analiz ederek, gelecek günün kapanış fiyatını tahmin eden bir **LSTM (Long Short-Term Memory)** modeli geliştirilmiştir.

Proje; veri toplama, ön işleme, model eğitimi, performans analizi ve son kullanıcı arayüzü (Web UI) aşamalarını kapsayan uçtan uca (end-to-end) bir makine öğrenmesi hattı (pipeline) sunar.

---

## 🚀 Özellikler

- **Canlı Veri Entegrasyonu:** `yfinance` kütüphanesi ile Yahoo Finance üzerinden güncel borsa verilerini otomatik çeker.
- **Gelişmiş Model Mimarisi:** Zaman serilerindeki uzun vadeli bağımlılıkları öğrenmek için LSTM ağları kullanılmıştır.
- **Veri Normalizasyonu:** Model başarımını artırmak için `MinMaxScaler` ile veriler ölçeklendirilmiştir.
- **Detaylı Görselleştirme:** Eğitim süreci, kayıp (loss) grafikleri ve tahmin başarısı için otomatik dashboard oluşturur.
- **İnteraktif Arayüz:** `Gradio` ile oluşturulan web arayüzü sayesinde, kullanıcılar manuel veri girerek anlık tahmin alabilirler.

---

## 🛠️ Kurulum

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin.

### 1. Repoyu Klonlayın

```bash
git clone https://github.com/KULLANICI_ADIN/REPO_ISMI.git
cd REPO_ISMI
```

### 2. Sanal Ortamı Oluşturun (Önerilen)

```bash
# Linux/Mac için
python3 -m venv venv
source venv/bin/activate

# Windows için
python -m venv venv
venv\Scripts\activate
```

### 3. Gerekli Kütüphaneleri Yükleyin

```bash
pip install -r requirements.txt
```

---

## ⚙️ Kullanım

Proje iki ana aşamadan oluşur: **Eğitim (Training)** ve **Sunum (Serving)**.

### Adım 1: Modeli Eğitme

Eğitim script'i veriyi indirir, işler, modeli eğitir ve sonuçları kaydeder.

```bash
# Eğer script kök dizindeyse:
python train.py

# Eğer script src/ altındaysa:
python src/train.py
```

Bu işlem tamamlandığında aşağıdaki çıktı dosyaları oluşturulur:

- `lstm_stock_model.pth` (model ağırlıkları)
- `scaler.save` (ölçekleyici)
- `detayli_analiz_dashboard.png` (analiz paneli)

### Adım 2: Arayüzü Başlatma

Eğitilen modeli kullanarak tahmin yapmak için web arayüzünü başlatın.

```bash
# Eğer script kök dizindeyse:
python serve.py

# Eğer script src/ altındaysa:
python src/serve.py
```

Terminalde verilen linke (örn: `http://127.0.0.1:7860`) tıklayarak tarayıcınızda demoyu görebilirsiniz.

---

## 📊 Model Performansı ve Sonuçlar

Model 200 epoch boyunca eğitilmiş ve test verisi üzerinde yüksek başarı göstermiştir.

- **Final Loss (MSE):** `0.0021`
- **Optimizasyon:** Adam
- **Mimari:** LSTM (Hidden Size: 50) + Linear Layer

### Analiz Grafikleri

Aşağıdaki grafik panelinde modelin eğitim süreci ve tahmin başarısı görülmektedir:

- **Zaman Serisi (Sol Üst):** Modelin tahminleri (Kırmızı), gerçek fiyatları (Mavi) başarıyla takip etmektedir.
- **Loss Eğrisi (Sağ Üst):** Hata oranı stabil bir şekilde düşmüştür.
- **Korelasyon (Sol Alt):** Tahmin ve gerçek değerler arasındaki yüksek korelasyon.
- **Hata Dağılımı (Sağ Alt):** Hataların sıfır noktası etrafında normal dağılım göstermesi.

---

## 📂 Proje Yapısı

```plaintext
├── data/                        # (Otomatik oluşur) Veri setleri
├── models/                      # Kaydedilen model dosyaları
├── src/
│   ├── model.py                 # LSTM Model Sınıfı (Mimarisi)
│   ├── train.py                 # Eğitim ve görselleştirme kodları
│   └── serve.py                 # Gradio arayüz kodları
├── lstm_stock_model.pth         # Eğitilmiş model ağırlıkları
├── scaler.save                  # Veri ölçekleyici (MinMax)
├── requirements.txt             # Bağımlılıklar
├── detayli_analiz_dashboard.png # Sonuç görseli
└── README.md                    # Proje dokümantasyonu
```

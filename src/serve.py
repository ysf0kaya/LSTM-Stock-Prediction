import gradio as gr
import torch
import numpy as np
import joblib
import os
from model import LSTMModelDemo

model_path = "../models/lstm_stock_model.pth"
scaler_path = "../models/scaler.save"

# Modeli ve Scaler'ı yükle
model = LSTMModelDemo()
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
else:
    print("Hata: Model dosyası bulunamadı! Lütfen önce eğitimi çalıştırın.")

model.eval()

# Scaler'ı yükle
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
else:
    print("Hata: Scaler dosyası bulunamadı!")

def process_input(input_str):
    try:
        # 1. Kullanıcı girdisini listeye çevir
        values = [float(x.strip()) for x in input_str.split(',')]
        
        if len(values) != 4:
            return "Hata: Lütfen tam olarak 4 adet fiyat giriniz."
        
        # 2. Veriyi modele uygun hale getir (Normalize et)
        # Scaler 2 boyutlu array bekler: (n_samples, n_features)
        values_array = np.array(values).reshape(-1, 1) 
        scaled_values = scaler.transform(values_array)
        
        # 3. Tensor'a çevir (Batch_size=1, Seq_len=4, Input_size=1)
        input_tensor = torch.tensor(scaled_values, dtype=torch.float32).view(1, 4, 1)
        
        # 4. Tahmin yap
        with torch.no_grad():
            prediction_scaled = model(input_tensor).item()
        
        # 5. Tahmini gerçek fiyata geri çevir (Inverse Transform)
        # prediction_scaled tek bir sayı, onu 2D array yapıp dönüştürüyoruz
        prediction_real = scaler.inverse_transform([[prediction_scaled]])[0][0]
        
        return f"Tahmini Sonraki Fiyat: {prediction_real:.2f}"
        
    except Exception as e:
        return f"Bir hata oluştu: {str(e)}"

# Arayüz
interface = gr.Interface(
    fn=process_input,
    inputs=gr.Textbox(lines=1, placeholder="Örn: 150.5, 151.2, 153.0, 152.8"),
    outputs="text",
    title="Borsa Fiyat Tahmin Modeli (LSTM)",
    description="Son 4 günün kapanış fiyatını girin, model 5. günü tahmin etsin.",
    examples=[
        ["175.50, 176.20, 178.00, 177.30"], # Örnek Apple fiyatları
        ["180.10, 179.50, 181.20, 182.00"]
    ]
)

if __name__ == "__main__":
    interface.launch(share=True)

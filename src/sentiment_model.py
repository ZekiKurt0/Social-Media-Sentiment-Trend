import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.preprocessing import clean_text  # Senin yazdığın temizlik fonksiyonu

class SentimentAnalyzer:
    def __init__(self, model_path="models/final_bert_model"):
        """
        BERT modelini ve Tokenizer'ı yerel klasörden yükler.
        """
        try:
            # Tokenizer ve Modeli yükle
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            
            # Modeli değerlendirme (inference) moduna al
            self.model.eval()
            
            # Etiket eşleşmeleri (LabelEncoder sırasına göre)
            self.labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
            print(f"✅ Model '{model_path}' konumundan başarıyla yüklendi.")
        except Exception as e:
            print(f"❌ Model yüklenirken hata oluştu: {e}")

    def predict(self, text):
        """
        Ham metni alır, temizler ve duygu tahmini yapar.
        """
        # 1. Metni temizle
        cleaned = clean_text(text)
        
        # 2. Metni BERT'in anlayacağı formata (token) çevir
        inputs = self.tokenizer(
            cleaned, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128
        )
        
        # 3. Gradyan hesaplamayı kapatıp tahmini yap
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Softmax ile olasılıkları hesapla
            # Denklemi: $$Softmax(x_i) = \frac{e^{x_i}}{\sum e^{x_j}}$$
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # En yüksek olasılıklı sınıfın indeksini al
            prediction_idx = torch.argmax(probabilities).item()
            
        return {
            "label": self.labels[prediction_idx],
            "confidence": float(probabilities[0][prediction_idx])
        }

# Dosyayı doğrudan çalıştırırsan test yapmanı sağlar
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    sample = "I hate waiting for my delayed flight!"
    print(f"Test Sonucu: {analyzer.predict(sample)}")
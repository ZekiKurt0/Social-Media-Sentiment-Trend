from src.sentiment_model import SentimentAnalyzer

def main():
    # Modeli yükle
    print("🤖 Model yükleniyor, lütfen bekleyin...")
    analyzer = SentimentAnalyzer(model_path="models/final_bert_model")
    
    print("\n--- 📝 Duygu Analizi Test Aracı ---")
    print("Çıkmak için 'exit' yazabilirsiniz.\n")

    while True:
        # Kullanıcıdan girdi al
        user_input = input("Cümlenizi girin: ")

        if user_input.lower() == 'exit':
            print("Görüşürüz! 👋")
            break

        if not user_input.strip():
            continue

        # Tahmin yap
        result = analyzer.predict(user_input)

        # Sonucu ekrana bas
        color = "🔴" if result['label'] == "Negative" else "🟢" if result['label'] == "Positive" else "🟡"
        print(f"{color} Sonuç: {result['label']} | Güven Oranı: %{result['confidence']*100:.2f}")
        print("-" * 40)

if __name__ == "__main__":
    main()
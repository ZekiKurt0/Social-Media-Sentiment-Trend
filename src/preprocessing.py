import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Küçük harfe çevir
    text = text.lower()
    # @kullanıcı adlarını kaldır
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    # URL'leri kaldır
    text = re.sub(r'http\S+', '', text)
    # Noktalama işaretlerini ve sayıları kaldır (sadece harfler kalsın)
    text = re.sub(r'[^a-z\s]', '', text)
    # Gereksiz boşlukları temizle ve stopwords'leri çıkar
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text.strip()
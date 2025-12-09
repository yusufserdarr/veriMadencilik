# Kahve Tüketimi ve Stres Analizi Projesi

Bu proje, öğrencilerin kahve tüketim alışkanlıkları ile stres düzeyleri arasındaki ilişkiyi analiz etmek amacıyla geliştirilmiştir. Proje kapsamında veriler toplanmış, temizlenmiş ve 3 farklı yapay zeka/makine öğrenmesi yöntemi ile analiz edilmiştir.

## 📂 Veri Seti Hakkında

Veri seti (`anket.csv`) öğrencilere yapılan bir anket sonucunda elde edilmiştir. İçerisinde şu bilgiler yer almaktadır:
*   **Yaş, Cinsiyet, İş/Okul Yoğunluğu**: Demografik ve yaşamsal bilgiler.
*   **Kahve Miktarı, Zamanı, Nedeni**: Kahve tüketim alışkanlıkları.
*   **Stres Düzeyi**: Hedef değişkenimiz (Düşük, Orta, Yüksek).
*   **Uyku Süresi, Ruh Hali**: Yan etkenler.

## 🛠 Kullanılan 3 Farklı Yöntem

Hocanın isteği üzerine veriler **3 farklı model** kullanılarak analiz edilmiştir:

1.  **K-En Yakın Komşu (KNN - K-Nearest Neighbors):** Benzer özelliklere sahip öğrencilerin stres düzeylerini gruplayarak tahmin eder.
2.  **Karar Ağacı (Decision Tree):** Veriyi sorularla dallara ayırarak (örn: "Günde 2'den fazla kahve içiyor mu?") bir karar ağacı oluşturur.
3.  **Rastgele Orman (Random Forest):** Birden fazla karar ağacını birleştirerek daha güçlü ve doğru tahminler yapar.

## 📊 Analiz Adımları

Proje şu adımları otomatik olarak gerçekleştirir:
1.  **Veri Temizleme:** Eksik veya hatalı veriler düzeltilir.
2.  **Korelasyon Analizi:** Hangi özelliğin stresle ne kadar ilgili olduğu incelenir.
3.  **Özellik Seçimi:** Modele katkısı olmayan gereksiz bilgiler çıkarılır.
4.  **Model Eğitimi:** Yukarıdaki 3 yöntem ile modeller eğitilir.
5.  **Karşılaştırma:** Hangi yöntemin en başarılı olduğu raporlanır.

## 🚀 Nasıl Çalıştırılır?

Analizi başlatmak için terminal veya komut satırında şu komutu yazmanız yeterlidir:

```bash
python3 kahve_stres_detayli_analiz.py
```

## 🏆 Örnek Sonuçlar

Analiz sonucunda genellikle **Rastgele Orman (Random Forest)** veya **Karar Ağacı** yöntemleri en yüksek başarıyı vermektedir. Kod çalıştığında size en iyi modeli ve başarı oranını (Örn: %60) söyleyecektir.

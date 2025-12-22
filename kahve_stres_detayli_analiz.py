import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("KAHVE TÜKETİMİ VE STRES İLİŞKİSİ - DETAYLI ANALİZ")
print("="*80)

# Veri yükleme
df = pd.read_csv('anket.csv')
print(f"\n✓ Veri yüklendi: {df.shape[0]} katılımcı, {df.shape[1]} sütun")

# Veri hazırlama
df_clean = df.drop('Zaman damgası', axis=1)
df_clean.columns = ['Yas', 'Cinsiyet', 'Is_Yogunlugu', 'Kahve_Miktar', 
                     'Kahve_Zamani', 'Kahve_Hissi', 'Stres_Duzeyi', 
                     'Uyku_Suresi', 'Ruh_Hali', 'Stresli_Kahve', 'Kahve_Nedeni']

# Kahve_Nedeni sütununu (Stresle başa çıkma vb. içerdiği için bias yaratıyor) çıkarıyoruz
if 'Kahve_Nedeni' in df_clean.columns:
    df_clean = df_clean.drop('Kahve_Nedeni', axis=1)
    print("⚠️ 'Kahve_Nedeni' sütunu bias önleme amacıyla çıkarıldı.")

def simplify(value):
    if pd.isna(value):
        return value
    if ',' in str(value):
        return str(value).split(',')[0].strip()
    return str(value)

df_ml = df_clean.copy()
for col in ['Kahve_Zamani', 'Kahve_Hissi', 'Uyku_Suresi', 'Ruh_Hali']:
    df_ml[col] = df_ml[col].apply(simplify)

# Label encoding
df_encoded = df_ml.copy()
for column in df_encoded.columns:
    if df_encoded[column].dtype == 'object':
        le = LabelEncoder()
        df_encoded[column] = le.fit_transform(df_encoded[column].astype(str))

print("\n" + "="*80)
print("ADIM 1: KORELASYON ANALİZİ")
print("="*80)

# Korelasyon analizi
correlation_matrix = df_encoded.corr()
stres_correlations = correlation_matrix['Stres_Duzeyi'].sort_values(ascending=False)

print("\n📊 Stres Düzeyi ile Diğer Değişkenler Arasındaki Korelasyon:")
print("-"*80)
for feature, corr_value in stres_correlations.items():
    if feature != 'Stres_Duzeyi':
        if abs(corr_value) >= 0.3:
            guc = "🔴 GÜÇLÜ"
        elif abs(corr_value) >= 0.15:
            guc = "🟡 ORTA "
        else:
            guc = "🟢 ZAYIF"
        print(f"{guc} - {feature:20s}: {corr_value:+.4f}")

print("\n" + "="*80)
print("ADIM 2: FEATURE IMPORTANCE ANALİZİ (Random Forest)")
print("="*80)

# Train-test split
X = df_encoded.drop('Stres_Duzeyi', axis=1)
y = df_encoded['Stres_Duzeyi']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Random Forest ile feature importance
rf_importance = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
rf_importance.fit(X_train, y_train)

feature_imp_df = pd.DataFrame({
    'Ozellik': X.columns,
    'Onem_Skoru': rf_importance.feature_importances_
}).sort_values('Onem_Skoru', ascending=False)

print("\n📊 Özellik Önem Skorları:")
print("-"*80)
for idx, row in feature_imp_df.iterrows():
    bar = '█' * int(row['Onem_Skoru'] * 100)
    if row['Onem_Skoru'] > 0.10:
        onem = "🔴 ÇOK ÖNEMLİ"
    elif row['Onem_Skoru'] > 0.05:
        onem = "🟡 ÖNEMLİ    "
    else:
        onem = "🟢 AZ ÖNEMLİ "
    print(f"{onem} - {row['Ozellik']:20s}: {row['Onem_Skoru']:.4f} {bar}")

print("\n" + "="*80)
print("ADIM 3: KORELASYON + FEATURE IMPORTANCE KARŞILAŞTIRMASI")
print("="*80)

# Karşılaştırma
comparison_df = pd.DataFrame({
    'Ozellik': X.columns,
    'Korelasyon': [stres_correlations[col] for col in X.columns],
    'Feature_Importance': rf_importance.feature_importances_
})
comparison_df['Abs_Korelasyon'] = comparison_df['Korelasyon'].abs()
comparison_df = comparison_df.sort_values('Feature_Importance', ascending=False)

print("\n📊 Her İki Analize Göre Özellik Değerlendirmesi:")
print("-"*80)
print(f"{'Özellik':<20} {'Korelasyon':>12} {'Feature Imp':>15} {'Karar':>15}")
print("-"*80)

onemli_ozellikler = []
onemsiz_ozellikler = []

for _, row in comparison_df.iterrows():
    ozellik = row['Ozellik']
    korr = row['Abs_Korelasyon']
    fi = row['Feature_Importance']
    
    if korr >= 0.10 or fi >= 0.05:
        karar = "✅ KULLAN"
        onemli_ozellikler.append(ozellik)
    else:
        karar = "❌ ÇIKAR"
        onemsiz_ozellikler.append(ozellik)
    
    print(f"{ozellik:<20} {row['Korelasyon']:>+12.4f} {fi:>15.4f} {karar:>15}")

print(f"\n✅ Kullanılacak özellik sayısı: {len(onemli_ozellikler)}")
if len(onemsiz_ozellikler) > 0:
    print(f"❌ Çıkarılacak özellikler: {', '.join(onemsiz_ozellikler)}")
    print(f"   (Düşük korelasyon ve feature importance nedeniyle)")

print("\n" + "="*80)
print("ADIM 4: SEÇİLEN ÖZELLİKLERLE MODEL EĞİTİMİ")
print("="*80)

# Seçilen özelliklerle yeni veri seti
X_selected = X[onemli_ozellikler]
X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nOrijinal özellik sayısı: {X.shape[1]}")
print(f"Seçilen özellik sayısı: {len(onemli_ozellikler)}")
print(f"Azaltma oranı: %{(1 - len(onemli_ozellikler)/X.shape[1])*100:.1f}")

# Model 1: KNN
print("\n1. K-NEAREST NEIGHBORS (KNN)")
print("-"*40)
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train_sel, y_train_sel)
knn_acc = accuracy_score(y_test_sel, knn.predict(X_test_sel))
print(f"   Doğruluk: %{knn_acc*100:.2f}")

# Model 2: Decision Tree
print("\n2. DECISION TREE")
print("-"*40)
dt = DecisionTreeClassifier(max_depth=8, random_state=42)
dt.fit(X_train_sel, y_train_sel)
dt_acc = accuracy_score(y_test_sel, dt.predict(X_test_sel))
print(f"   Doğruluk: %{dt_acc*100:.2f}")

# Model 3: Random Forest
print("\n3. RANDOM FOREST")
print("-"*40)
rf = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
rf.fit(X_train_sel, y_train_sel)
rf_acc = accuracy_score(y_test_sel, rf.predict(X_test_sel))
print(f"   Doğruluk: %{rf_acc*100:.2f}")

# Model 4: Naive Bayes
print("\n4. NAIVE BAYES (GaussianNB)")
print("-"*40)
nb_model = GaussianNB()
nb_model.fit(X_train_sel, y_train_sel)
nb_acc = accuracy_score(y_test_sel, nb_model.predict(X_test_sel))
print(f"   Doğruluk: %{nb_acc*100:.2f}")

# En iyi model
models = [('KNN', knn, knn_acc), ('Decision Tree', dt, dt_acc), ('Random Forest', rf, rf_acc), ('Naive Bayes', nb_model, nb_acc)]
best_model = max(models, key=lambda x: x[2])

print("\n" + "="*80)
print("SONUÇLAR")
print("="*80)

print("\n📊 MODEL PERFORMANSLARI (Detaylı):")
print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
print("-" * 65)

for model_name, model, acc in sorted(models, key=lambda x: x[2], reverse=True):
    y_pred = model.predict(X_test_sel)
    # Zero division handling for safety
    prec = precision_score(y_test_sel, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test_sel, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test_sel, y_pred, average='weighted', zero_division=0)
    
    star = "🏆 " if model_name == best_model[0] else "   "
    print(f"{star}{model_name:20s}: %{acc*100:.2f}      {prec:.4f}     {rec:.4f}     {f1:.4f}")

print(f"\n🏆 EN İYİ MODEL: {best_model[0]} (%{best_model[2]*100:.2f})")

print("\n💡 ÖNEMLI BULGULAR:")
print(f"  1. En önemli özellikler: {', '.join(onemli_ozellikler[:3])}")
if len(onemsiz_ozellikler) > 0:
    print(f"  2. Çıkarılan özellikler: {', '.join(onemsiz_ozellikler)}")
print(f"  3. Veri boyutu optimizasyonu: %{(1 - len(onemli_ozellikler)/X.shape[1])*100:.1f} azaltma")
print(f"  4. Model başarısı: Üç yöntem de %{min([m[2] for m in models])*100:.1f} - %{max([m[2] for m in models])*100:.1f} doğruluk aralığında")

print("\n✅ HOCANIN İSTEDİĞİ ANALİZLER TAMAMLANDI:")
print("  ✓ Korelasyon Analizi")
print("  ✓ Feature Importance Analizi")
print("  ✓ Anlamsız özelliklerin tespiti ve çıkarılması")
print("  ✓ Üç farklı makine öğrenmesi modeli (KNN, DT, RF, NB)")
print("  ✓ Model performans karşılaştırması")

print("\n" + "="*80)


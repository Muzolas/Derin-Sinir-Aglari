import numpy as np
import pickle
import os

# -------------------------------------------------------------------------
# ADIM 1: VERİ SETİNİ YEREL DOSYADAN YÜKLEME
# -------------------------------------------------------------------------

# Dosya yolu
data_folder = 'data/cifar-10-python/cifar-10-batches-py'

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Sadece ilk eğitim paketini (10.000 resim) ve test paketini yüklüyoruz (Hız için)
train_batch = unpickle(os.path.join(data_folder, 'data_batch_1'))
test_batch = unpickle(os.path.join(data_folder, 'test_batch'))

X_train = train_batch[b'data'].astype("float32")
y_train = np.array(train_batch[b'labels'])

X_test = test_batch[b'data'].astype("float32")
y_test = np.array(test_batch[b'labels'])

# İşlem süresini kısaltmak için eğitim setinden ilk 5000, testten ilk 10 örneği alalım
X_train = X_train[:5000]
y_train = y_train[:5000]
X_test = X_test[:10]
y_test = y_test[:10]

# -------------------------------------------------------------------------
# ADIM 2: KULLANICI GİRİŞLERİ
# Ödev kuralı: Önce mesafe seçimi, sonra k değeri alınacak.
# -------------------------------------------------------------------------

print("--- CIFAR-10 k-NN Sınıflandırıcı ---")
print("1. L1 (Manhattan)")
print("2. L2 (Öklid)")

secim = input("Lütfen mesafe türünü seçin (L1 veya L2): ").strip().upper()
k_degeri = int(input("Lütfen k değerini (komşu sayısı) giriniz: "))

# -------------------------------------------------------------------------
# ADIM 3: SINIFLANDIRMA (DÜZ AKIŞ)
# Ödev kuralı: Fonksiyonlara atlamadan, kod açık ve düz şekilde çalışacak.
# -------------------------------------------------------------------------

dogru_tahmin = 0

print("\nSınıflandırma işlemi başlıyor...\n")

# Test setindeki her bir resim için döngü
for i in range(len(X_test)):
    test_resmi = X_test[i]
    gercek_etiket = y_test[i]

    # MESAFE HESAPLAMA
    if secim == "L1":
        # Manhattan: Mutlak farkların toplamı
        mesafeler = np.sum(np.abs(X_train - test_resmi), axis=1)
    elif secim == "L2":
        # Öklid: Farkların karelerinin toplamının karekökü
        mesafeler = np.sqrt(np.sum(np.square(X_train - test_resmi), axis=1))
    else:
        print("Hatalı seçim yaptınız!")
        break

    # EN YAKIN KOMŞULARI BULMA
    # Mesafeleri küçükten büyüğe sıralayıp en yakın 'k' tanesinin indeksini alıyoruz
    en_yakin_indeksler = np.argsort(mesafeler)[:k_degeri]

    # Bu indekslere karşılık gelen etiketleri (sınıfları) alıyoruz
    en_yakin_etiketler = y_train[en_yakin_indeksler]

    # OYLAMA (Tahmin Yapma)
    # En çok tekrar eden etiketi buluyoruz
    tahmin = np.bincount(en_yakin_etiketler).argmax()

    # SONUÇLARI YAZDIRMA
    durum = "DOĞRU" if tahmin == gercek_etiket else "YANLIŞ"
    if tahmin == gercek_etiket:
        dogru_tahmin += 1

    print(f"Test Örneği {i + 1}: Tahmin: {tahmin}, Gerçek: {gercek_etiket} [{durum}]")

# Genel Başarı Oranı
basari = (dogru_tahmin / len(X_test)) * 100
print(f"\nToplam Başarı: %{basari}")
print("-" * 35)
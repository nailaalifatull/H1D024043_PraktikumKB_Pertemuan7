## Identitas Praktikan
* **Nama:** Naila Alifatul Mabruroh
* **NIM:** H1D024043
* **Shift:** D

---

## Deskripsi Program
Program ini melakukan proses *end-to-end* pembelajaran mesin (Machine Learning), mulai dari ekstraksi dataset, pra-pemrosesan data, pembuatan arsitektur jaringan saraf, pelatihan model, evaluasi performa, hingga prediksi data baru secara interaktif.

Dataset yang digunakan adalah **Iris Dataset** dari Scikit-Learn yang terdiri dari 150 sampel data dari 3 spesies bunga Iris: *Setosa*, *Versicolor*, dan *Virginica*. Setiap data memiliki 4 fitur penentu:
1. **Sepal Length** (Panjang Kelopak)
2. **Sepal Width** (Lebar Kelopak)
3. **Petal Length** (Panjang Mahkota)
4. **Petal Width** (Lebar Mahkota)

---

## Penjelasan Struktur Kode

Kode program dibagi menjadi 12 tahapan utama yang terstruktur secara berurutan:

### 1. Inisialisasi Library (Importing Libraries)
Mengimpor semua modul yang diperlukan:
* `tensorflow` & `keras` untuk membangun dan melatih model Jaringan Saraf Tiruan.
* `pandas` & `numpy` untuk manipulasi matriks dan struktur data.
* `sklearn` untuk memuat dataset, melakukan pengkodean label, membagi data, dan menghitung matriks kebingungan (*confusion matrix*).
* `matplotlib` & `seaborn` untuk visualisasi grafik performa dan matriks evaluasi (menggunakan backend `'Agg'` agar grafik langsung disimpan ke file tanpa membuka jendela *pop-up*).

### 2. Memuat Dataset (Load Dataset)
Memanggil dataset Iris bawaan melalui `load_iris()`. Fitur numerik disimpan dalam variabel `X` dan target kelas disimpan dalam variabel `y`.

### 3. Transformasi Label (Label Encoding)
Mengubah nama target spesies (teks) menjadi bentuk numerik kontinu ($0$, $1$, dan $2$) menggunakan `LabelEncoder`. Hal ini penting agar label dapat diproses secara optimal oleh loss function pada Jaringan Saraf.

### 4. Pembagian Data (Data Splitting)
Memisahkan dataset menjadi dua bagian menggunakan `train_test_split`:
* **Data Latih (Training Set):** 80% dari total data, digunakan untuk melatih bobot model.
* **Data Validasi / Uji (Testing Set):** 20% dari total data, digunakan untuk menguji performa model pada data yang belum pernah dipelajari sebelumnya.
* `random_state=42` diset agar hasil pembagian data konsisten setiap kali program dijalankan.

### 5. Arsitektur Model Jaringan Saraf
Model dibangun menggunakan struktur *Sequential* dengan layer-layer sebagai berikut:
* **Input Layer:** Menerima input dengan dimensi sesuai jumlah fitur data latih (4 fitur).
* **Hidden Layer 1:** 1000 neuron, fungsi aktivasi `ReLU`.
* **Hidden Layer 2:** 500 neuron, fungsi aktivasi `ReLU`.
* **Hidden Layer 3:** 300 neuron, fungsi aktivasi `ReLU`.
* **Output Layer:** 3 neuron (mewakili kelas *Setosa*, *Versicolor*, *Virginica*), menggunakan fungsi aktivasi `Softmax` untuk menghasilkan distribusi probabilitas kelas.

### 6. Kompilasi Model (Compile Model)
Mengonfigurasi proses pembelajaran model:
* **Optimizer:** `adam` (Adaptive Moment Estimation) untuk pembaruan bobot yang efisien.
* **Loss Function:** `sparse_categorical_crossentropy` karena label target berupa integer (bukan *one-hot encoded*).
* **Metrics:** `accuracy` untuk memantau persentase tebakan model yang benar.

### 7. Pelatihan Model (Model Training)
Melatih model menggunakan `model.fit()` sebanyak **50 Epoch** (iterasi global) dengan ukuran *batch* (sampel per proses) sebesar **32**. Data uji juga dimasukkan sebagai `validation_data` untuk memantau gejala *overfitting* selama pelatihan.

### 8. Evaluasi Performa
Menghitung nilai *Loss* dan *Accuracy* akhir menggunakan data uji (`X_test` dan `y_test`) untuk mengetahui performa riil model.

### 9. Visualisasi Tren Latihan (Training History Plot)
Menyimpan riwayat akurasi dan loss dari objek `history` ke dalam file gambar `training_history.png`. Grafik ini membantu menganalisis apakah model mengalami *underfitting* atau *overfitting*.

### 10. Prediksi Data Uji
Model melakukan prediksi terhadap seluruh data uji. Kelas dengan nilai probabilitas tertinggi diambil menggunakan `.argmax(axis=1)` lalu dicetak berdampingan dengan label asli untuk perbandingan langsung.

### 11. Matriks Kebingungan (Confusion Matrix Heatmap)
Membuat diagram *Confusion Matrix* menggunakan `seaborn.heatmap()` untuk melihat detail akurasi klasifikasi per spesies secara visual. Hasil visualisasi disimpan otomatis sebagai `confusion_matrix.png`.

### 12. Prediksi Interaktif Data Baru
Menyediakan fungsi dinamis `predict_new_data()` yang memungkinkan pengguna menginputkan nilai sepal dan petal secara manual di terminal, melakukan prediksi instan, dan mengembalikan nama spesies asli hasil konversi `inverse_transform`.

---

## File Output yang Dihasilkan
Setelah program berhasil dijalankan, direktori kerja Anda akan memiliki dua file gambar baru:
1. `training_history.png` – Grafik garis perkembangan akurasi dan loss per epoch.
2. `confusion_matrix.png` – Matriks evaluasi klasifikasi benar vs salah untuk tiap spesies bunga Iris.

---

## Cara Menjalankan Program
python index.py

# Laporan Proyek Machine Learning - Taufan Fajarama PR

## Domain Proyek: Social Science

Sistem *bike-sharing* adalah generasi baru dari penyewaan sepeda tradisional di mana seluruh proses mulai dari keanggotaan, penyewaan, hingga pengembalian menjadi otomatis. Melalui sistem ini, pengguna dapat dengan mudah menyewa sepeda dari satu lokasi dan mengembalikannya di lokasi lain. Saat ini, terdapat lebih dari 500 program *bike-sharing* di seluruh dunia dengan lebih dari 500 ribu sepeda. Sistem ini sangat diminati karena perannya yang penting dalam masalah lalu lintas, lingkungan, dan kesehatan.

Masalah mobilitas pengguna *bike-sharing* penting untuk diselesaikan karena dapat membantu dalam perencanaan dan pengelolaan sumber daya yang lebih baik untuk sistem *bike-sharing*. Dengan prediksi yang akurat, pihak pengelola dapat memastikan ketersediaan sepeda di berbagai lokasi dan meningkatkan pengalaman pengguna.

Referensi penelitian terkait:
- [Event labeling combining ensemble detectors and background knowledge (Fanaee-T & Gama, 2013)](https://www.semanticscholar.org/paper/Event-labeling-combining-ensemble-detectors-and-Fanaee-T-Gama/bc42899f599d31a5d759f3e0a3ea8b52479d6423)
  > Menggunakan machine learning untuk melabeli event/kejadian besar yang berkaitan dengan data bike-sharing
- [Recurrent Neural Networks for Time Series Forecasting (Petnehàzi, 2019)](https://www.semanticscholar.org/paper/Recurrent-Neural-Networks-for-Time-Series-Petneh%C3%A1zi/ed4a2a2ed51cc7418c2d1ca8967cc7a383c0241a)
  > Menggunakan machine learning untuk melakukan time series forecasting pada data bike-sharing

## Business Understanding

### Problem Statements
- Kapan waktu atau musim puncak (*peak hour/season*) pengguna sepeda biasa (`casual`) dan langganan (`registered`)?
- Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh terhadap jumlah pengguna sepeda biasa (`casual`) dan langganan (`registered`)?
- Berapa jumlah pengguna sepeda biasa (`casual`) dan langganan (`registered`) pada kondisi karakteristik tertentu?

### Goals
- Mengetahui waktu atau musim puncak (*peak hour/season*) pengguna sepeda biasa (`casual`) dan langganan (`registered`).
- Mengetahui fitur yang paling berkorelasi dengan jumlah pengguna sepeda biasa (`casual`) dan langganan (`registered`).
- Membuat model machine learning yang dapat memprediksi jumlah pengguna sepeda biasa (`casual`) dan langganan (`registered`) seakurat mungkin berdasarkan karakteristik atau fitur-fitur yang ada.

### Solution Statements
- Melakukan Time Series Analysis dan Multivariate Analysis dari pengguna sepeda biasa (`casual`) dan langganan (`registered`) pada dataset.
- Menggunakan dua atau lebih algoritma machine learning untuk mencapai solusi yang diinginkan, seperti algoritma K-Nearest Neighbor, Random Forest, Adaptive Boosting, dan Gradient Boosting untuk membangun model prediksi.
- Mengevaluasi model dengan metrik yang sesuai, seperti Mean Squared Error (MSE) dan memilih model terbaik berdasarkan hasil evaluasi.

## Data Understanding

Dataset yang digunakan (`hour.csv` dan `day.csv`) berisi jumlah sepeda sewa per jam dan harian antara tahun 2011 dan 2012 dalam sistem Capital Bikeshare, dengan informasi cuaca dan musim yang sesuai. Dataset diambil dari: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset).

### Variabel atau Fitur pada Dataset
`hour.csv` dan `day.csv` sama-sama memiliki kolom berikut, kecuali `hr` tidak ada di `day.csv`

- `instant`: record index
- `dteday`: date
- `season`: musim (1: musim dingin, 2: musim semi, 3: musim panas, 4: musim gugur)
- `yr`: year (0: 2011, 1: 2012)
- `mnth`: month (1-12)
- `hr`: hour (0-23)
- `holiday`: apakah hari libur atau bukan
- `weekday`: hari dalam minggu (0: Minggu, 1: Senin, dst.)
- `workingday`: apakah hari kerja atau bukan
- `weathersit`: kondisi cuaca (1: cerah, 2: mendung, 3: hujan ringan, 4: hujan deras)
- `temp`: suhu
- `atemp`: suhu terasa
- `hum`: kelembaban
- `windspeed`: kecepatan angin
- `casual`: jumlah pengguna casual
- `registered`: jumlah pengguna terdaftar
- `cnt`: jumlah total pengguna

### Exploratory Data Analysis
- Pada `day.csv` dapat diamati bahwa:
  - Terdapat 1 kolom dengan tipe object, yaitu: `dteday`. Kolom ini menandakan time-series data.
  - Terdapat 11 kolom dengan tipe data int64. Terdiri dari 1 kolom indeks (`instant`), 5 kolom categorical features yang sudah di-encoding dalam numerik (`season, yr, mnth, weekday, weathersit`) dan binary (`holiday, workingday`), dan 3 kolom target (`casual, registered, cnt`).
  - Terdapat 4 kolom dengan tipe data float64. Terdiri dari 4 kolom numerical features (`temp, atemp, hum, windspeed`).
- Pada `hour.csv` dapat diamati bahwa:
  - Sama seperti `day.csv` dengan tambahan 1 kolom bertipe data int64, yaitu kolom `hr`, categorical features yang sudah di-encoding dalam numerik.
- Tidak terdapat missing value pada dataset `day` dan dataset `hour`
- Pada categorical features, kolom `season`, `yr`, `mnth`, `hr`, `holiday`, `weekday`, dan `workingday` memiliki nilai unik yang konsisten untuk kedua dataset. Sedangkan kolom `weathersit` terdapat perbedaan di mana dataset `day` hanya memiliki 3 nilai dan dataset `hour` memiliki 4 nilai.
- Pada numerical features, kolom `temp` dan `atemp` tidak terdapat outlier di kedua dataset. Sedangkan kolom `hum` dan `windspeed` terdapat outlier di kedua dataset.
- Informasi pada histogram kolom `casual`, sebagian data terdapat pada jumlah pengguna perjam di bawah 50 dengan distribusi data miring ke kanan (right-skewed). 
- Informasi pada histogram kolom `registered`, sebagian data terdapat pada jumlah pengguna perjam di bawah 200 dengan distribusi data miring ke kanan (right-skewed).
- Pengguna `casual` dan `registered` menunjukkan kenaikan jumlah di tahun 2012 dibanding 2011.
- Pengguna `casual` cenderung lebih banyak di sekitar jam 12-16, sedangkan pengguna `registered` puncaknya di jam 8 dan jam 17-18. Dapat diasumsikan bahwa pengguna `registered` adalah para pekerja yang berangkat kerja (jam 8) dan pulang kerja (jam 17).
- Pada pola sebaran pairplot dan correlation matrix untuk fitur numerik dataset `day` dan `hour` tidak terdapat fitur yang signifikan berkorelasi (mendekati -1 atau 1) terhadap kolom target (`casual` dan `registered`). 
  - Kolom `temp` dan `atemp` memiliki korelasi paling besar pada dataset `day` dengan nilai korelasi 0.54.
  - Kolom `temp` memiliki korelasi paling besar terhadap kedua target pada dataset `hour` dengan nilai korelasi 0.46 dan 0.34.

## Data Preparation

### Transformasi Dataset
Berdasarkan informasi atribut dan informasi awal dataset `day.csv` dan `hour.csv`, diputuskan untuk melakukan beberapa transformasi:
- menghapus kolom `instant` karena hanya menunjukkan indeks
- merubah datatype kolom `dteday` menjadi datetime dan menjadikannya indeks dataframe
- merubah categorical features `season`, `weekday`, dan `weathersit` dari bentuk numerikal menjadi ordinal agar lebih deskriptif dan nantinya dapat dilakukan one hot encoding
- menghapus kolom `cnt` sehingga kolom target hanya `casual` dan `registered` karena kolom `cnt` hanya menjumlahkan kedua kolom tersebut

### Penanganan Outliers
Kedua dataset merupakan time-series dataset sehingga apabila ada baris yang dihapus akan menghilangkan kontinuitas dari data.

Maka dari itu, outlier di kedua dataset akan dilakukan imputasi nilai menggunakan nilai `lower_bound` (Q1 - 1.5 * IQR) atau `upper_bound` (Q3 + 1.5 * IQR) sehingga kontinuitas data tetap terjaga dan outlier dapat ditangani.

### Encoding Fitur Kategori
Fitur kategorikal pada kolom `season`, `mnth`, `hr`, `weekday`, dan `weathersit` dilakukan one-hot encoding menggunakan `pd.get_dummies` sehingga kolom fitur kategorikal tersebut dapat menjadi vector binary yang lebih sesuai untuk algoritma machine learning.

### Reduksi Dimensi dengan PCA
Pada pola sebaran pairplot dan correlation matrix di section EDA, dapat dilihat bahwa kolom `temp` dan `atemp` berkorelasi tinggi. Kolom `temp` dan `atemp` memiliki korelasi yang tinggi satu sama lain menunjukkan data yang berulang atau redundant sehingga dapat direduksi dimensinya. Kedua kolom direduksi menjadi 1 kolom `temperature` menggunakan PCA (Principal Component Analysis).

### Train-Test-Split
train_test_split dataset `day` menggunakan proporsi pembagian data latih dan uji 80:20 karena memiliki jumlah sampel kurang dari 1.000 sampel. Sedangkan train_test_split dataset `hour` menggunakan proporsi pembagian data latih dan uji 90:10 karena memiliki jumlah sampel yang banyak (lebih dari 10.000 sampel). Kolom target `casual` dan `registered` dipisah untuk mempermudah analisis prediksi yang modular.

### Standarisasi Fitur Numerik
Standarisasi fitur numerik pada kolom `temperature`, `hum`, dan `windspeed` dilakukan menggunakan `StandardScaler` untuk menggeser distribusi data mendekati distribusi normal. Algoritma machine learning cenderung memiliki performa yang lebih baik ketika data memiliki skala relatif sama. Standarisasi dilakukan untuk data latih terlebih dahulu, nantinya akan dilakukan juga untuk data uji sebelum evaluasi.


## Modeling

Model yang digunakan untuk menyelesaikan permasalahan akan mengikuti yang diajarkan di kelas (K-Nearest Neighbor, Random Forest, Adaptive Boosting) dengan tambahan 1 algoritma gradient boosting, yaitu algoritma XGBoost (eXtreme Gradient Boosting). Seluruh algoritma dilatih dan diuji pada empat dataset yang berbeda: `day` dengan target `casual`, `day` dengan target `registered`, `hour` dengan target `casual`, dan `hour` dengan target `registered`. Berikut adalah penjelasan untuk setiap algoritma yang digunakan:

- K-Nearest Neighbor (KNN):
  - Tahapan:
    - Memilih nilai K (jumlah tetangga terdekat).
    - Kemudian, menghitung jarak antara data yang akan diregresi dengan semua data lainnya.
    - Memilih K tetangga terdekat berdasarkan jarak tersebut.
    - Lalu, menentukan label mayoritas dari K tetangga tersebut sebagai label prediksi.
  - Parameter:
    - Model KNN dilatih dengan nilai K (jumlah tetangga terdekat) sebesar 10 dan metric Euclidean untuk mengukur jarak antara titik seperti di kelas.
  - Kelebihan:
    - Algoritma ini sederhana dan mudah diimplementasikan.
    - Cocok untuk data dengan fitur yang relatif sedikit.
  - Kekurangan:
    - Sensitif terhadap data pencilan (outliers).

- Random Forest:
  - Tahapan:
    - Membentuk ensemble (group) dari beberapa pohon keputusan.
    - Setiap pohon dibangun dengan subset data dan fitur acak.
    - Prediksi akhir didapatkan dengan menggabungkan hasil dari semua pohon.
  - Parameter:
   - Model Random Forest dilatih dengan 50 estimator, kedalaman maksimum 16, menggunakan semua core processor (n_jobs=-1), dan random state 55 untuk mengontrol random number generator yang digunakan.
  - Kelebihan:
    - Cocok untuk klasifikasi dan regresi.
    - Tahan terhadap overfitting.
  - Kekurangan:
    - Membutuhkan lebih banyak sumber daya komputasi.

- Adaptive Boosting (AdaBoost):
  - Tahapan:
    - Inisialisasi bobot sama untuk semua instance.
    - Melatih model dan menghitung error
    - Menyesuaikan bobot pada observasi yang salah diklasifikasikan.
    - Menggabungkan model dengan bobot baru dengan proses iteratif dilakukan hingga konvergen.
  - Parameter:
    - Model AdaBoost dilatih dengan learning rate sebesar 0.05 sebagai bobot yang diterapkan pada setiap regressoor di masing-masing proses iterasi boosting dan random state 55 untuk mengontrol random number generator yang digunakan.
  - Kelebihan:
    - Meningkatkan akurasi secara adaptif.
    - Bekerja dengan baik pada pohon yang pendek dan sederhana.
  - Kekurangan:
    - Bisa menjadi tidak stabil dengan data yang bervariasi.

- Gradient Boosting (XGBoost):
  - Tahapan:
    - Inisialisasi model dengan prediksi dasar.
    - Proses iteratif dilakukan dengan mengurangi residual error antara prediksi model dan nilai aktual.
    - Melatih model baru untuk memprediksi residual (pada XGBoost, menggunakan regularisasi).
    - Gradien digunakan untuk memperbaiki model.
    - Memperbarui prediksi dengan menambahkan model baru ke model yang ada.
  - Parameter:
    - Model XGBoost dilatih dengan learning rate sebesar 0.05 sebagai bobot yang diterapkan pada setiap regressoor di masing-masing proses iterasi boosting dan random state 55 untuk mengontrol random number generator yang digunakan.
  - Kelebihan:
    - Performa yang luar biasa dan skalabilitas yang baik.
    - Menyediakan regularisasi untuk mencegah overfitting.
  - Kekurangan:
    - Memerlukan tuning parameter yang lebih ekstensif.

- Keempat algoritma machine learning akan dilatih dan diuji, dengan asumsi awal model terbaik yang dapat dipilih adalah Gradient Boosting (XGBoost) karena performanya yang luar biasa dan regularisasi yang dilakukan dapat mencegah overfitting.


## Evaluation

Metrik evaluasi yang digunakan adalah Mean Squared Error (MSE) karena model yang dilakukan adalah regresi. Pada kasus regresi, jika prediksi mendekati nilai sebenarnya (eror kecil), performanya baik. Sedangkan jika tidak (eror besar), performanya buruk. Metrik MSE mengukur rata-rata kuadrat kesalahan atau deviasi antara nilai prediksi dan nilai sebenarnya.

Formula MSE: 
$$MSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - {y\_pred_i})^2 \$$

Keterangan:

$N =$ jumlah dataset

$yi =$ nilai sebenarnya

$y\_pred_i =$ nilai prediksi

MSE dihitung untuk setiap baris prediksi dengan nilai yang sebenarnya, dalam proyek ini, nilai `casual` dan `registered` dari dataset `day` dan `hour` yang menjadi nilai untuk diprediksi dan dihitung erornya. Semakin rendah nilai MSE, maka model memberikan performa yang semakin baik.

### Hasil Evaluasi
- MSE pada data train dan test dihitung untuk masing-masing model.

#### Hasil Proyek dengan Metrik MSE pada Dataset `day` dengan Target Kolom `casual`
  
|              | train      |	test       |
| ------------ | ---------- | ---------- |
| KNN	         | 113.323376 | 79.738204  |
| RandomForest |	12.236806 |	84.088612  |
| AdaBoost     |	98.04128  |	125.703101 |
| XGBoost      |	6.283987  |	78.384786  |

- Dapat dilihat bahwa untuk dataset `day` dengan target `casual`, model XGBoost memberikan nilai error (MSE) yang paling kecil daripada algoritma lain pada data latih dan data uji, sehingga model XGBoost akan digunakan untuk melakukan prediksi jumlah pengguna `casual` pada dataset `day`.

#### Hasil Proyek dengan Metrik MSE pada Dataset `day` dengan Target Kolom `registered`
  
|              | train      |	test       |
| ------------ | ---------- | ---------- |
| KNN	         | 551.825814 | 713.40972  |
| RandomForest |	47.369535 |	377.075951 |
| AdaBoost     | 530.839485 |	675.899531 |
| XGBoost      |	30.385332 |	385.472429 |

- Dapat dilihat bahwa untuk dataset `day` dengan target `registered`, model XGBoost memberikan nilai error (MSE) yang paling kecil daripada algoritma lain pada data latih. Akan tetapi, model RandomForest memberikan nilai error (MSE) yang paling kecil daripada algoritma lain pada data uji, sehingga model RandomForest akan digunakan untuk melakukan prediksi jumlah pengguna `registered` pada dataset `day`.

#### Hasil Proyek dengan Metrik MSE pada Dataset `hour` dengan Target Kolom `casual`
  
|              | train    |	test     |
| ------------ | -------- | -------- |
| KNN	         | 0.455017 | 0.582879 |
| RandomForest | 0.126084 |	0.438576 |
| AdaBoost     | 1.192262 |	1.275039 |
| XGBoost      | 0.338983 |	0.475651 |

- Dapat dilihat bahwa untuk dataset `hour` dengan target `casual`, model RandomForest memberikan nilai error (MSE) yang paling kecil daripada algoritma lain pada data latih dan data uji, sehingga model RandomForest akan digunakan untuk melakukan prediksi jumlah pengguna `casual` pada dataset `hour`.

#### Hasil Proyek dengan Metrik MSE pada Dataset `hour` dengan Target Kolom `registered`
  
|              | train     |	test     |
| ------------ | --------- | --------- |
| KNN	         | 7.455045  | 9.076764  |
| RandomForest | 1.383121  | 2.687725  |
| AdaBoost     | 12.167595 | 12.034242 |
| XGBoost      | 2.570123  | 2.959357  |

- Dapat dilihat bahwa untuk dataset `hour` dengan target `registered`, model RandomForest memberikan nilai error (MSE) yang paling kecil daripada algoritma lain pada data latih dan data uji, sehingga model RandomForest akan digunakan untuk melakukan prediksi jumlah pengguna `registered` pada dataset `hour`.


### Kesimpulan
- Dari keempat kasus, yaitu kombinasi dataset `day` dan `hour` dengan kolom target `casual` dan `registered`, model RandomForest memberikan hasil yang paling baik pada 3 dari 4 kasus uji.
- Jawaban dari business understanding lainnya dapat dilihat di bagian [Exploratory Data Analysis](#exploratory-data-analysis)
- Untuk melihat visualisasi dan pemahaman yang lebih lengkap dapat dilihat di source code `ipynb` atau `py`.

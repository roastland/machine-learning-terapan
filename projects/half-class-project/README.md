# Laporan Proyek Machine Learning - Taufan Fajarama PR

## Domain Proyek: *Social Science*

Sistem *bike-sharing* adalah generasi baru dari penyewaan sepeda tradisional di mana seluruh proses mulai dari keanggotaan, penyewaan, hingga pengembalian menjadi otomatis. Melalui sistem ini, pengguna dapat dengan mudah menyewa sepeda dari satu lokasi dan mengembalikannya di lokasi lain. Saat ini, terdapat lebih dari 500 program *bike-sharing* di seluruh dunia dengan lebih dari 500 ribu sepeda. Sistem *bike-sharing* sangat diminati karena perannya yang penting dalam masalah lalu lintas, lingkungan, dan kesehatan [[1](#referensi)].

### Latar Belakang Proyek

Dengan semakin banyaknya sistem *bike-sharing*, jumlah pengguna terus meningkat dan pada saat-saat tertentu ketersediaan sepeda bisa menjadi masalah. Oleh karena itu, pola mobilitas pengguna *bike-sharing* perlu diidentifikasi dan diatasi agar dapat membantu dalam perencanaan dan pengelolaan sumber daya yang lebih baik untuk sistem ini. 

Dengan prediksi yang akurat, pihak pengelola dapat memastikan ketersediaan sepeda pada waktu atau musim puncak di berbagai lokasi, sehingga dapat meningkatkan pengalaman pengguna. Pentingnya pola ini adalah untuk memastikan bahwa sepeda tersedia pada saat dibutuhkan, menghindari kekurangan sepeda yang bisa menyebabkan ketidaknyamanan pengguna, serta untuk meningkatkan efisiensi operasional dan kepuasan pelanggan.

Masalah utama yang dihadapi adalah mengidentifikasi waktu atau musim puncak penggunaan sepeda, menentukan kondisi yang paling memengaruhi jumlah pengguna, dan memprediksi jumlah pengguna berdasarkan karakteristik tertentu. Dengan memahami pola ini, pengelola dapat mengoptimalkan distribusi sepeda, mengurangi ketidaknyamanan pengguna karena kekurangan sepeda, dan meningkatkan efisiensi operasional serta kepuasan pelanggan.

Model *machine learning* dapat memproses data historis untuk menemukan pola penggunaan, memberikan prediksi yang akurat mengenai permintaan di masa depan, dan membantu pengambilan keputusan yang lebih baik dalam pengelolaan *bike-sharing*. Dengan demikian, penggunaan model *machine learning* dapat memberikan solusi yang signifikan dalam manajemen sistem *bike-sharing* ini [[2](#referensi)].

Referensi penelitian terkait:
- [Fanaee-T & Gama, 2013](https://www.semanticscholar.org/paper/Event-labeling-combining-ensemble-detectors-and-Fanaee-T-Gama/bc42899f599d31a5d759f3e0a3ea8b52479d6423)
- [Petnehàzi, 2019](https://www.semanticscholar.org/paper/Recurrent-Neural-Networks-for-Time-Series-Petneh%C3%A1zi/ed4a2a2ed51cc7418c2d1ca8967cc7a383c0241a)

## *Business Understanding*

Penelitian ini bermanfaat untuk:
- Pengelola *bike-sharing* dalam pengelolaan inventaris.
- Pembuat kebijakan dalam perencanaan infrastruktur.
- Peningkatan pengalaman pengguna.

Dengan *stakeholders* yang dituju pada hasil akhir penelitian adalah pengelola *bike-sharing*, pengguna, dan pembuat kebijakan transportasi.

### *Problem Statements*
1. Bagaimana cara mengidentifikasi waktu atau musim puncak pengguna sepeda biasa (*casual*) dan langganan (*registered*)?
2. Bagaimana menentukan fitur yang paling berpengaruh terhadap jumlah pengguna sepeda biasa (*casual*) dan langganan (*registered*)?
3. Bagaimana memprediksi jumlah pengguna sepeda biasa (*casual*) dan langganan (*registered*) berdasarkan karakteristik tertentu?

### *Goals*
1. Mengetahui waktu atau musim puncak pengguna sepeda biasa dan langganan.
2. Mengetahui fitur yang paling berkorelasi dengan jumlah pengguna sepeda biasa dan langganan.
3. Membuat model *machine learning* untuk memprediksi jumlah pengguna sepeda biasa dan langganan berdasarkan karakteristik yang ada.

### *Solution Statements*
- Melakukan *Time Series Analysis* dan *Multivariate Analysis* dari pengguna sepeda biasa (`casual`) dan langganan (`registered`) pada *dataset*.
- Menggunakan dua atau lebih algoritma *machine learning* untuk mencapai solusi yang diinginkan, seperti algoritma *K-Nearest Neighbor*, *Random Forest*, *Adaptive Boosting*, dan *Gradient Boosting* untuk membangun model prediksi.
- Mengevaluasi model dengan metrik yang sesuai, seperti *Mean Squared Error* (MSE) dan memilih model terbaik berdasarkan hasil evaluasi.

## *Data Understanding*

*Dataset* yang digunakan (`hour.csv` dan `day.csv`) berisi jumlah sepeda sewa per jam dan harian antara tahun 2011 dan 2012 dalam sistem Capital Bikeshare, dengan informasi cuaca dan musim yang sesuai. Dataset diambil dari: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset).

### Informasi *Dataset*
`day.csv` memiliki 731 jumlah *records* dengan 16 kolom fitur dan target. `hour.csv` memiliki 17379 jumlah *records* dengan 17 kolom fitur dan target. `hour.csv` dan `day.csv` sama-sama memiliki kolom berikut (kecuali `hr`, tidak ada di `day.csv`):

- `instant`: indeks *record*, bertipe data `int64`, kolom ini menandakan indeks data;
- `dteday`: tanggal, bertipe data `object`, kolom ini menandakan *time-series* data;
- `season`: musim (1: musim dingin, 2: musim semi, 3: musim panas, 4: musim gugur), bertipe data `int64`, kolom ini menandakan *categorical features* yang sudah di-*encoding* dalam numerik;
- `yr`: tahun (0: 2011, 1: 2012), bertipe data `int64`, kolom ini menandakan *categorical features* yang sudah di-*encoding* dalam numerik;
- `mnth`: bulan (1-12), bertipe data `int64`, kolom ini menandakan *categorical features* yang sudah di-*encoding* dalam numerik;
- `hr`: jam (0-23), bertipe data `int64`, kolom ini menandakan *categorical features* yang sudah di-*encoding* dalam numerik;
- `holiday`: apakah hari libur atau bukan, bertipe data `int64`, kolom ini menandakan *categorical features* yang sudah di-*encoding* dalam biner;
- `weekday`: hari dalam minggu (0: Minggu, 1: Senin, dst.), bertipe data `int64`, kolom ini menandakan *categorical features* yang sudah di-*encoding* dalam numerik;
- `workingday`: apakah hari kerja atau bukan, bertipe data `int64`, kolom ini menandakan *categorical features* yang sudah di-*encoding* dalam biner;
- `weathersit`: kondisi cuaca (1: cerah, 2: mendung, 3: hujan ringan, 4: hujan deras), bertipe data `int64`, kolom ini menandakan *categorical features* yang sudah di-*encoding* dalam numerik;
- `temp`: suhu, bertipe data `float64`, kolom ini menandakan *numerical features*;
- `atemp`: suhu terasa, bertipe data `float64`, kolom ini menandakan *numerical features*;
- `hum`: kelembaban, bertipe data `float64`, kolom ini menandakan *numerical features*;
- `windspeed`: kecepatan angin, bertipe data `float64`, kolom ini menandakan *numerical features*;
- `casual`: jumlah pengguna biasa, bertipe data `int64`, kolom ini menandakan kolom target;
- `registered`: jumlah pengguna terdaftar, bertipe data `int64`, kolom ini menandakan kolom target;
- `cnt`: jumlah total pengguna, bertipe data `int64`, kolom ini menandakan kolom target.

### *Exploratory Data Analysis* (EDA)
- Tidak terdapat *missing value* pada *dataset* `day` dan *dataset* `hour`
- Pada *categorical features*, kolom `season`, `yr`, `mnth`, `hr`, `holiday`, `weekday`, dan `workingday` memiliki nilai unik yang konsisten untuk kedua *dataset*. Sedangkan kolom `weathersit` terdapat perbedaan di mana *dataset* `day` hanya memiliki 3 nilai dan *dataset* `hour` memiliki 4 nilai.
- Pada *boxplot numerical features* di Gambar 1 dan Gambar 2, kolom `temp` dan `atemp` tidak terdapat *outlier* di kedua *dataset*. Sedangkan kolom `hum` dan `windspeed` terdapat *outlier* di kedua *dataset*.
  
![Boxplot Day](https://github.com/roastland/machine-learning-terapan/blob/main/projects/half-class-project/assets/1.png?raw=true)

*Gambar 1: Boxplot fitur numerikal pada dataset `day`*

![Boxplot Hour](https://github.com/roastland/machine-learning-terapan/blob/main/projects/half-class-project/assets/2.png?raw=true)

*Gambar 2: Boxplot fitur numerikal pada dataset `hour`*

- Pada Gambar 3, terlihat pengguna `casual` dan `registered` menunjukkan kenaikan jumlah di tahun 2012 dibanding 2011.

![Time-Series Line Chart](https://github.com/roastland/machine-learning-terapan/blob/main/projects/half-class-project/assets/3.png?raw=true)

*Gambar 3: Time-Series Line Chart pengguna `casual` dan `registered`*

- Pada Gambar 4, terlihat pengguna `casual` cenderung lebih banyak di sekitar jam 12-17, sedangkan pengguna `registered` puncaknya di jam 8 dan jam 17-18. Dapat diasumsikan bahwa pengguna `registered` adalah para pekerja yang berangkat kerja (jam 8) dan pulang kerja (jam 17).

![Hourly Line Chart](https://github.com/roastland/machine-learning-terapan/blob/main/projects/half-class-project/assets/4.png?raw=true)

*Gambar 4: Hourly Line Chart pengguna `casual` dan `registered`*

- Pada Gambar 5 dan Gambar 6, terlihat pengguna `casual` cenderung lebih banyak di hari Sabtu-Minggu (*weekend*), sedangkan pengguna `registered` lebih banyak di hari Senin-Jumat (*weekday*). Dapat diasumsikan bahwa pengguna `casual` adalah mayoritas pengguna yang sedang liburan dan pengguna `registered` adalah mayoritas para pekerja kantoran.

![Bar Chart Casual](https://github.com/roastland/machine-learning-terapan/blob/main/projects/half-class-project/assets/5.png?raw=true)

*Gambar 5: Bar Chart pengguna `casual` setiap harinya*

![Bar Chart Registered](https://github.com/roastland/machine-learning-terapan/blob/main/projects/half-class-project/assets/6.png?raw=true)

*Gambar 6: Bar Chart pengguna `registered` setiap harinya*

- Pada *correlation matrix* untuk fitur numerik *dataset* `day` di Gambar 7 dan *dataset* `hour` di Gambar 8 tidak terdapat fitur yang signifikan berkorelasi (mendekati -1 atau 1) terhadap kolom target (`casual` dan `registered`). Akan tetapi:
  - Kolom `temp` dan `atemp` memiliki korelasi paling besar pada *dataset* `day` dengan nilai korelasi 0.54 terhadap target `casual` dan `registered`.
  - Kolom `temp` memiliki korelasi paling besar pada *dataset* `hour` dengan nilai korelasi 0.46 terhadap target `casual` dan 0.34 terhadap target `registered`.

![Corr Matrix Day](https://github.com/roastland/machine-learning-terapan/blob/main/projects/half-class-project/assets/7.png?raw=true)

*Gambar 7: Correlation Matrix fitur numerik dataset `day`*

![Corr Matrix Hour](https://github.com/roastland/machine-learning-terapan/blob/main/projects/half-class-project/assets/8.png?raw=true)

*Gambar 8: Correlation Matrix fitur numerik dataset `hour`*

EDA lebih lengkap dapat dilihat di [notebook](https://github.com/roastland/machine-learning-terapan/blob/main/projects/half-class-project/submission_1.ipynb).

## *Data Preparation*

### Transformasi *Dataset*
Berdasarkan informasi atribut dan informasi awal *dataset* `day.csv` dan `hour.csv`, diputuskan untuk melakukan beberapa transformasi:
- menghapus kolom `instant` karena hanya menunjukkan indeks
- merubah *datatype* kolom `dteday` menjadi *datetime* dan menjadikannya indeks *dataframe*
- merubah *categorical features* `season`, `weekday`, dan `weathersit` dari bentuk numerikal menjadi ordinal agar lebih deskriptif dan nantinya dapat dilakukan *one hot encoding*
- menghapus kolom `cnt` sehingga kolom target hanya `casual` dan `registered` karena kolom `cnt` hanya menjumlahkan kedua kolom tersebut

### Penanganan *Outliers*
Kedua *dataset* merupakan *time-series dataset* sehingga apabila ada baris yang dihapus akan menghilangkan kontinuitas dari data.

Maka dari itu, *outlier* di kedua *dataset* akan dilakukan imputasi nilai menggunakan nilai `lower_bound` $(Q1 - 1.5 * IQR)$ atau `upper_bound` $(Q3 + 1.5 * IQR)$ sehingga kontinuitas data tetap terjaga dan *outlier* dapat ditangani.

### *Encoding* Fitur Kategori
Fitur kategorikal pada kolom `season`, `mnth`, `hr`, `weekday`, dan `weathersit` dilakukan *one-hot encoding* menggunakan `pd.get_dummies` sehingga kolom fitur kategorikal tersebut dapat menjadi *binary vector* yang lebih sesuai untuk algoritma *machine learning*.

### Reduksi Dimensi dengan PCA
Pada `correlation matrix` di section [EDA](#exploratory-data-analysis), dapat dilihat bahwa kolom `temp` dan `atemp` berkorelasi tinggi. Kolom `temp` dan `atemp` memiliki korelasi yang tinggi satu sama lain menunjukkan data yang berulang atau *redundant* sehingga dapat direduksi dimensinya. Kedua kolom direduksi menjadi 1 kolom `temperature` menggunakan PCA (*Principal Component Analysis*).

### *Train-Test-Split*
- Pada `train_test_split` *dataset* `day` digunakan proporsi pembagian data latih dan uji `80:20` karena memiliki jumlah sampel kurang dari 1.000 sampel. 
- Sedangkan `train_test_split` *dataset* `hour` digunakan proporsi pembagian data latih dan uji `90:10` karena memiliki jumlah sampel yang banyak (lebih dari 10.000 sampel). 
- Kolom target `casual` dan `registered` dipisah untuk mempermudah analisis prediksi yang modular dan lebih menjawab [*business understanding*](#business-understanding).

### Standarisasi Fitur Numerik
Standarisasi fitur numerik pada kolom `temperature`, `hum`, dan `windspeed` dilakukan menggunakan `StandardScaler` untuk menggeser distribusi data mendekati distribusi normal. Algoritma *machine learning* cenderung memiliki performa yang lebih baik ketika data memiliki skala relatif sama. Standarisasi dilakukan untuk data latih terlebih dahulu, nantinya akan dilakukan juga untuk data uji sebelum evaluasi.


## *Modeling*

Model yang digunakan untuk menyelesaikan permasalahan akan mengikuti yang diajarkan di kelas (*K-Nearest Neighbor, Random Forest, Adaptive Boosting*) dengan tambahan 1 algoritma *gradient boosting*, yaitu algoritma *XGBoost* (*eXtreme Gradient Boosting*). Seluruh algoritma dilatih dan diuji pada empat *dataset* yang berbeda: `day` dengan target `casual`, `day` dengan target `registered`, `hour` dengan target `casual`, dan `hour` dengan target `registered`. Berikut adalah penjelasan untuk setiap algoritma yang digunakan:

- *K-Nearest Neighbor* (KNN):
  - Tahapan:
    - Memilih nilai K (jumlah tetangga terdekat).
    - Kemudian, menghitung jarak antara data yang akan diregresi dengan semua data lainnya.
    - Memilih K tetangga terdekat berdasarkan jarak tersebut.
    - Lalu, menentukan label mayoritas dari K tetangga tersebut sebagai label prediksi.
  - Parameter:
    - Model KNN dilatih dengan nilai K (jumlah tetangga terdekat) sebesar 10 dan metric *Euclidean* untuk mengukur jarak antara titik seperti di kelas.
  - Kelebihan:
    - Algoritma ini sederhana dan mudah diimplementasikan.
    - Cocok untuk data dengan fitur yang relatif sedikit.
  - Kekurangan:
    - Sensitif terhadap data pencilan (*outliers*).

- *Random Forest*:
  - Tahapan:
    - Membentuk *ensemble* (*group*) dari beberapa pohon keputusan.
    - Setiap pohon dibangun dengan *subset* data dan fitur acak.
    - Prediksi akhir didapatkan dengan menggabungkan hasil dari semua pohon.
  - Parameter:
    - Model *Random Forest* dilatih dengan 50 *estimator*, kedalaman maksimum 16, menggunakan semua *core processor* (`n_jobs=-1`), dan *random state* 55 untuk mengontrol *random number generator* yang digunakan.
  - Kelebihan:
    - Cocok untuk klasifikasi dan regresi.
    - Tahan terhadap *overfitting*.
  - Kekurangan:
    - Membutuhkan lebih banyak sumber daya komputasi.

- *Adaptive Boosting* (AdaBoost):
  - Tahapan:
    - Inisialisasi bobot sama untuk semua *instance*.
    - Melatih model dan menghitung eror.
    - Menyesuaikan bobot pada observasi yang salah diklasifikasikan.
    - Menggabungkan model dengan bobot baru yang dilakukan dengan proses iteratif hingga konvergen.
  - Parameter:
    - Model AdaBoost dilatih dengan *learning rate* sebesar 0.05 sebagai bobot yang diterapkan pada setiap *regressor* di masing-masing proses iterasi *boosting* dan *random state* 55 untuk mengontrol *random number generator* yang digunakan.
  - Kelebihan:
    - Meningkatkan akurasi secara adaptif.
    - Bekerja dengan baik pada pohon yang pendek dan sederhana.
  - Kekurangan:
    - Bisa menjadi tidak stabil dengan data yang bervariasi.

- *Gradient Boosting* (XGBoost):
  - Tahapan:
    - Inisialisasi model dengan prediksi dasar.
    - Proses iteratif dilakukan dengan mengurangi residual eror antara prediksi model dan nilai aktual.
    - Melatih model baru untuk memprediksi residual (pada XGBoost, menggunakan regularisasi).
    - Gradien digunakan untuk memperbaiki model.
    - Memperbarui prediksi dengan menambahkan model baru ke model yang ada.
  - Parameter:
    - Model XGBoost dilatih dengan *learning rate* sebesar 0.05 sebagai bobot yang diterapkan pada setiap *regressor* di masing-masing proses iterasi *boosting* dan *random state* 55 untuk mengontrol *random number generator* yang digunakan.
  - Kelebihan:
    - Performa yang luar biasa dan skalabilitas yang baik.
    - Menyediakan regularisasi untuk mencegah *overfitting*.
  - Kekurangan:
    - Memerlukan *tuning* parameter yang lebih ekstensif.

Keempat algoritma *machine learning* akan dilatih dan diuji, dengan asumsi awal model terbaik yang dapat dipilih adalah *Gradient Boosting* (XGBoost) karena performanya yang luar biasa dan regularisasi yang dilakukan dapat mencegah *overfitting*.


## *Evaluation*

### Metrik Evaluasi
Metrik evaluasi yang digunakan adalah *Mean Squared Error* (MSE) karena model yang dilakukan adalah regresi. Pada kasus regresi, jika prediksi mendekati nilai sebenarnya (eror kecil), performanya baik. Sedangkan jika tidak (eror besar), performanya buruk. Metrik MSE mengukur rata-rata kuadrat kesalahan atau deviasi antara nilai prediksi dan nilai sebenarnya.

Formula MSE: 
$$MSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - {y\\_pred_i})^2$$

Keterangan:

$N =$ jumlah dataset

$y_i =$ nilai sebenarnya

$y\\_pred_i =$ nilai prediksi

MSE dihitung untuk setiap baris prediksi dengan nilai yang sebenarnya, dalam proyek ini, nilai `casual` dan `registered` dari dataset `day` dan `hour` yang menjadi nilai untuk diprediksi dan dihitung erornya. Semakin rendah nilai MSE, maka model memberikan performa yang semakin baik.

### Hasil Evaluasi
Sebelum model dievaluasi, data uji dilakukan standarisasi skala fitur numerikal yang sama dengan data latih menggunakan `StandardScaler`. 

Hasil evaluasi setiap model algoritma menggunakan metrik MSE pada Dataset `day` dengan Target `casual` dapat dilihat di Tabel 1 dan Gambar 9.

*Tabel 1. Hasil Evaluasi MSE pada Dataset `day` dengan Target `casual`*
|              | train      |	test       |
| ------------ | ---------- | ---------- |
| KNN	         | 113.323376 | 79.738204  |
| RandomForest |	12.236806 |	84.088612  |
| AdaBoost     |	98.04128  |	125.703101 |
| XGBoost      |	6.283987  |	78.384786  |

![Rank Model 1](https://github.com/roastland/machine-learning-terapan/blob/main/projects/half-class-project/assets/9.png?raw=true)

*Gambar 9: Ranking model berdasarkan MSE pada data uji Dataset `day` dengan Target `casual`*

Hasil evaluasi setiap model algoritma menggunakan metrik MSE pada Dataset `day` dengan Target `registered` dapat dilihat di Tabel 2 dan Gambar 10.

*Tabel 2. Hasil Evaluasi MSE pada Dataset `day` dengan Target `registered`*
|              | train      |	test       |
| ------------ | ---------- | ---------- |
| KNN	         | 551.825814 | 713.40972  |
| RandomForest |	47.369535 |	377.075951 |
| AdaBoost     | 530.839485 |	675.899531 |
| XGBoost      |	30.385332 |	385.472429 |

![Rank Model 2](https://github.com/roastland/machine-learning-terapan/blob/main/projects/half-class-project/assets/10.png?raw=true)

*Gambar 10: Ranking model berdasarkan MSE pada data uji Dataset `day` dengan Target `registered`*

Hasil evaluasi setiap model algoritma menggunakan metrik MSE pada Dataset `hour` dengan Target `casual` dapat dilihat di Tabel 3 dan Gambar 11.

*Tabel 3. Hasil Evaluasi MSE pada Dataset `hour` dengan Target `casual`*
|              | train    |	test     |
| ------------ | -------- | -------- |
| KNN	         | 0.455017 | 0.582879 |
| RandomForest | 0.126084 |	0.438576 |
| AdaBoost     | 1.192262 |	1.275039 |
| XGBoost      | 0.338983 |	0.475651 |

![Rank Model 3](https://github.com/roastland/machine-learning-terapan/blob/main/projects/half-class-project/assets/11.png?raw=true)

*Gambar 11: Ranking model berdasarkan MSE pada data uji Dataset `hour` dengan Target `casual`*

Hasil evaluasi setiap model algoritma menggunakan metrik MSE pada Dataset `hour` dengan Target `registered` dapat dilihat di Tabel 4 dan Gambar 12.

*Tabel 4. Hasil Evaluasi MSE pada Dataset `hour` dengan Target `registered`*
|              | train     |	test     |
| ------------ | --------- | --------- |
| KNN	         | 7.455045  | 9.076764  |
| RandomForest | 1.383121  | 2.687725  |
| AdaBoost     | 12.167595 | 12.034242 |
| XGBoost      | 2.570123  | 2.959357  |

![Rank Model 4](https://github.com/roastland/machine-learning-terapan/blob/main/projects/half-class-project/assets/12.png?raw=true)

*Gambar 12: Ranking model berdasarkan MSE pada data uji Dataset `hour` dengan Target `registered`*

### Kesimpulan
- Model `XGBoost` memberikan hasil terbaik (nilai MSE/eror terkecil) pada *dataset* `day` dengan target `casual`.
- Model `RandomForest` memberikan hasil terbaik (nilai MSE/eror terkecil) pada *dataset* `day` dengan target `registered` dan pada *dataset* `hour` dengan target `casual` dan `registered`.
- Model `RandomForest` lebih konsisten memberikan hasil terbaik (nilai MSE/eror terkecil) pada tiga dari empat kasus uji.
- Proyek berhasil menjawab [*business understanding*](#business-understanding):
  - Proyek berhasil menghasilkan model *machine learning* untuk memprediksi jumlah pengguna sepeda biasa dan langganan berdasarkan karakteristik yang ada.
  - Hasil evaluasi model dengan metrik *Mean Squared Error* (MSE) sudah sesuai dan dapat dipilih model terbaik berdasarkan hasil evaluasi, yaitu model `RandomForest`.
  - Dapat diketahui fitur yang paling berkorelasi (fitur numerik `temp` dan `atemp`) dengan jumlah pengguna sepeda biasa dan langganan serta waktu atau musim puncak *demand*-nya (pengguna biasa `casual` pada masa liburan di sekitar jam 12-17, pengguna terdaftar `registered` pada hari kerja di jam 8 dan jam 17-18).

## Referensi

[[1] Fanaee-T, H., & Gama, J. (2013). Event labeling combining ensemble detectors and background knowledge. *Progress in Artificial Intelligence*, 13-25.](https://www.semanticscholar.org/paper/Event-labeling-combining-ensemble-detectors-and-Fanaee-T-Gama/bc42899f599d31a5d759f3e0a3ea8b52479d6423)

[[2] Petnehàzi, G. (2019). Recurrent Neural Networks for Time Series Forecasting. *ArXiv preprint*, arXiv:1901.00069.](https://www.semanticscholar.org/paper/Recurrent-Neural-Networks-for-Time-Series-Petneh%C3%A1zi/ed4a2a2ed51cc7418c2d1ca8967cc7a383c0241a)

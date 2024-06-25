# Laporan Proyek Machine Learning - Taufan Fajarama PR

## Project Overview 

Rekomendasi buku merupakan bagian penting dalam meningkatkan pengalaman membaca pengguna dan meningkatkan penjualan buku. Dengan meningkatnya jumlah buku yang diterbitkan setiap tahun, pengguna sering kesulitan menemukan buku yang sesuai dengan minat dan preferensi mereka. Sistem rekomendasi buku otomatis menjadi solusi untuk masalah ini, memungkinkan pengguna menemukan buku baru yang mungkin mereka sukai berdasarkan riwayat dan preferensi mereka [[1](#referensi)][[2](#referensi)].

Masalah utama yang dihadapi adalah bagaimana mengidentifikasi buku-buku yang relevan untuk direkomendasikan kepada pengguna berdasarkan data historis. Dataset Book-Crossing yang dikumpulkan oleh Cai-Nicolas Ziegler dalam pengumpulan data selama 4 minggu (Agustus / September 2004) dari komunitas Book-Crossing dengan izin dari Ron Hornbaker, CTO Humankind Systems, menyediakan informasi tentang pengguna, buku, dan rating yang diberikan, yang dapat digunakan untuk membangun model rekomendasi.

Dengan menggunakan model *machine learning*, seperti model *content-based filtering* atau *collaborative filtering*, kita dapat menganalisis data ini untuk menemukan pola preferensi pengguna dan memberikan rekomendasi yang lebih akurat dan personal. Penerapan sistem rekomendasi ini tidak hanya meningkatkan kepuasan pengguna tetapi juga dapat meningkatkan penjualan buku dan keterlibatan pengguna dalam platform.

Referensi penelitian terkait:
- [Mathew et al., 2016](https://ieeexplore.ieee.org/abstract/document/7684166)
- [Ng, 2020](https://www.inderscienceonline.com/doi/abs/10.1504/IJBIDM.2020.104738)


## Business Understanding

Penelitian ini bermanfaat untuk:
- Penerbit dan penjual buku dalam meningkatkan penjualan.
- Pengguna untuk menemukan buku yang sesuai dengan minat mereka.

Dengan *stakeholders* yang dituju pada hasil akhir penelitian adalah penerbit, penjual buku, dan pengguna.

### Problem Statements
- Berdasarkan data mengenai pengguna, bagaimana membuat sistem rekomendasi yang dipersonalisasi dengan teknik *content-based filtering*?
- Dengan data rating yang dimiliki, bagaimana komunitas dapat merekomendasikan buku lain yang mungkin disukai dan belum pernah dibaca oleh pengguna?

### Goals
- Menghasilkan sejumlah rekomendasi buku yang dipersonalisasi untuk pengguna dengan teknik *content-based filtering*.
- Menghasilkan sejumlah rekomendasi buku yang sesuai dengan preferensi pengguna dan belum pernah dibaca sebelumnya dengan teknik *collaborative filtering*.

### Solution Statements
- Melakukan analisis data pengguna dan buku dari dataset *Book-Crossing*.
- Menggunakan dua atau lebih algoritma *machine learning* untuk membangun model rekomendasi, seperti *Collaborative Filtering* dan *Content-Based Filtering*.


## *Data Understanding*

*Dataset* yang digunakan (`Ratings.csv`, `Books.csv`, dan `Users.csv`) berisi informasi tentang pengguna, buku, dan rating yang diberikan oleh pengguna pada buku-buku yang berbeda. Dataset ini dikumpulkan oleh Cai-Nicolas Ziegler dari komunitas Book-Crossing dengan izin dari Ron Hornbaker, CTO Humankind Systems, dan mencakup periode pengumpulan data selama 4 minggu (Agustus / September 2004). Dataset diambil dari: [Kaggle](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset).

### Informasi *Dataset*

#### Users (`Users.csv`):
Dataset pengguna mengandung informasi demografis dan anonim dari pengguna Book-Crossing. Beberapa pengguna mungkin memiliki informasi seperti lokasi dan usia. Dataset ini memiliki 278,858 entri pengguna, di mana beberapa pengguna mungkin memiliki informasi usia yang tidak lengkap.

**Informasi Kolom**:
- `User-ID`: ID pengguna, bertipe data `int64`.
- `Location`: Lokasi pengguna, bertipe data `object`.
- `Age`: Usia pengguna, bertipe data `float64`.


#### Books (`Books.csv`):
Dataset buku mengidentifikasi buku berdasarkan ISBN. Informasi tambahan meliputi judul buku, pengarang, tahun publikasi, penerbit, dan URL yang mengarah ke gambar sampul buku dari Amazon Web Services. Dataset ini memiliki 271,360 entri buku dengan beberapa kolom memiliki nilai null.

**Informasi Kolom**:
- `ISBN`: Nomor ISBN buku, bertipe data `object`.
- `Book-Title`: Judul buku, bertipe data `object`.
- `Book-Author`: Pengarang buku, bertipe data `object`.
- `Year-Of-Publication`: Tahun publikasi buku, bertipe data `object`.
- `Publisher`: Penerbit buku, bertipe data `object`.
- `Image-URL-S`, `Image-URL-M`, `Image-URL-L`: URL gambar sampul buku dalam tiga ukuran yang berbeda, bertipe data `object`.


#### Ratings (`Ratings.csv`):
Dataset rating buku berisi informasi tentang rating yang diberikan oleh pengguna pada buku tertentu. Rating dapat bersifat eksplisit, diwakili dalam skala 1-10, atau implisit, diwakili oleh nilai 0 jika tidak ada rating eksplisit yang diberikan. Dataset ini memiliki 1,149,780 entri rating buku yang mencakup kedua jenis rating, eksplisit dan implisit.

**Informasi Kolom**:
- `User-ID`: ID pengguna yang memberikan rating, bertipe data `int64`.
- `ISBN`: Nomor ISBN buku yang diberi rating, bertipe data `object`.
- `Book-Rating`: Rating buku yang diberikan oleh pengguna, bertipe data `int64`.


### *Exploratory Data Analysis* (EDA)

- Pada Gambar 1, dari 278858 pengguna di dataset `users`, hanya ada 105283 user di dataset `ratings` yang menandakan tidak semua pengguna pernah menilai suatu buku. Hal ini bisa menguatkan alasan untuk mengembangkan sistem rekomendasi kepada pengguna yang belum pernah membaca suatu buku.

![Boxplot Day](https://github.com/roastland/machine-learning-terapan/blob/main/projects/final-project/assets/1.png?raw=true)

*Gambar 1: Nilai unik di tiap dataset*

- Pada Gambar 2 dan 3, dapat dilihat bahwa terdapat kesalahan pada data buku di kolom Year-Of-Publication yang seharusnya adalah tahun, hal ini akan diperbaiki pada bagian selanjutnya. 

![Kesalahan pada data buku](https://github.com/roastland/machine-learning-terapan/blob/main/projects/final-project/assets/2.png?raw=true)

*Gambar 2: Kesalahan pada data buku*

![Nilai Tahun Publikasi yang salah](https://github.com/roastland/machine-learning-terapan/blob/main/projects/final-project/assets/3.png?raw=true)

*Gambar 3: Nilai Tahun Publikasi yang salah*

- Pada Gambar 4, dapat dilihat bahwa distribusi tahun publikasi merupakan *left-skewed*, sehingga jika terdapat data yang invalid akan diisi dengan nilai median.

![Histogram Tahun Publikasi](https://github.com/roastland/machine-learning-terapan/blob/main/projects/final-project/assets/4.png?raw=true)

*Gambar 4: Histogram Tahun Publikasi*

## Data Preparation

### Transformasi *Dataset*
Berdasarkan informasi atribut dan EDA pada *dataset*, diputuskan untuk melakukan beberapa transformasi:
- menghapus 3 kolom Image pada data buku karena tidak diperlukan pada proyek ini
- menghapus kolom usia dan lokasi pengguna pada data pengguna karena tidak diperlukan pada proyek ini
- menangani data tahun publikasi yang salah karena seharusnya kolom tahun hanya diisi int, selain itu untuk data tahun bernilai 0 atau di atas 2006, akan diubah dengan median karena bersifat invalid (dataset penelitian dilakukan tahun 2004, 2 tahun tambahan untuk antisipasi jika dataset aslinya terdapat perubahan)
- menghapus data pada dataset `ratings` yang bernilai 0 (rating implisit) karena pada proyek ini hanya memanfaatkan rating eksplisit.

### Persiapan Data untuk *Content-Based Filtering*
- ketiga dataset digabungkan menjadi satu untuk mempermudah analisis
- kemudian kolom penulis buku, tahun publikasi, dan *publisher* akan digabungkan menjadi satu kolom fitur buku seperti pada Gambar 5, hal ini dilakukan untuk memperkaya konteks *similarity*

![Penggabungan Fitur](https://github.com/roastland/machine-learning-terapan/blob/main/projects/final-project/assets/5.png?raw=true)

*Gambar 5: Penggabungan Fitur*

- kemudian data yang memiliki nilai null akan dihapus dan sisa data yang ada dimasukkan ke dictionary pada Gambar 6 untuk mempermudah melakukan *content-based filtering*

![Dictionary Content Based](https://github.com/roastland/machine-learning-terapan/blob/main/projects/final-project/assets/6.png?raw=true)

*Gambar 6: Dictionary Content Based*

### Persiapan Data untuk *Collaborative Filtering*
- Pada Gambar 7 dilakukan mapping data pengguna dan buku untuk setiap ratingnya untuk mempermudah *collaborative filtering*

![Mapping user dan buku](https://github.com/roastland/machine-learning-terapan/blob/main/projects/final-project/assets/7.png?raw=true)

*Gambar 7: Mapping tiap user dan tiap buku kepada ratingnya*


## Modeling
Tahapan dan output model *machine learning* dapat dilihat lebih lengkap di [notebook](https://github.com/roastland/machine-learning-terapan/blob/main/projects/final-project/submission_2.ipynb).

### *Content-Based Filtering*
Model ini memiliki kelebihan jika data interaksi pengguna tidak mencukupi karena model ini berfokus pada konten dari item yang akan direkomendasikan. Akan tetapi, kekurangannya adalah kemungkinan tidak dapat merekomendasikan item yang sesuai jika fitur yang dimiliki kurang lengkap.

Maka dari itu model ini pada proyek rekomendasi buku ini akan menggabungkan fitur yang akan untuk memperkaya konteksnya. Metode vektorisasi fiturnya akan menggunakan `CountVectorizer` karena setelah dicoba dengan `TF-IDF` tidak dapat memberikan hasil yang optimal

Data yang digunakan merupakan 1000 sampel dari seluruh data untuk meringankan beban komputasi saat menghitung `cosine similarity`.

### *Collaborative Filtering*
Model ini memiliki kelebihan jika data historis interaksi pengguna cukup representatif karena model ini berfokus pada histori interaksi pengguna dengan item yang akan direkomendasikan. Akan tetapi, kekurangannya adalah kemungkinan tidak dapat merekomendasikan item yang sesuai jika data historis yang dimiliki kurang mencukupi.

Model ini pada proyek rekomendasi buku ini akan menggunakan `layer` yang disediakan `Keras` dengan beberapa modifikasi seperti di kelas. Data yang digunakan merupakan 1000 sampel dari seluruh data untuk meringankan beban komputasi saat melatih model.

## Evaluation
### Metrik Evaluasi
Metrik evaluasi yang digunakan adalah *Root Mean Squared Error* (RMSE) karena model yang dilakukan adalah prediksi kemiripan vektor. Jika prediksi mendekati nilai sebenarnya (eror kecil), performanya baik. Sedangkan jika tidak (eror besar), performanya buruk. Metrik RMSE mengukur akar dari rata-rata kuadrat kesalahan atau deviasi antara nilai prediksi dan nilai sebenarnya.

Formula MSE: 
$$RMSE = \sqrt{ \frac{1}{N} \sum_{i=1}^{N} (y_i - {y\\_pred_i})^2}$$

Keterangan:

$N =$ jumlah dataset

$y_i =$ nilai sebenarnya

$y\\_pred_i =$ nilai prediksi

RMSE dihitung untuk setiap baris prediksi dengan nilai yang sebenarnya, dalam proyek ini, nilai `similarity vector` yang menjadi nilai untuk diprediksi dan dihitung erornya. Semakin rendah nilai RMSE, maka model memberikan performa yang semakin baik.

### Hasil Evaluasi
Sebelum model dievaluasi, data uji dilakukan diambil hanya 1000 sampel untuk pengujian karena data yang dimiliki sangat besar dan memakan beban komputasi yang besar, sehingga hanya diambil sampel yang cukup untuk melihat performanya. Hasil RMSE pada Gambar 8 menunjukkan performa model cukup baik karena hasil eror yang tidak terlalu besar.

![Hasil RMSE](https://github.com/roastland/machine-learning-terapan/blob/main/projects/final-project/assets/8.png?raw=true)

*Gambar 8: Hasil RMSE*


## Kesimpulan
Proyek berhasil menjawab [*business understanding*](#business-understanding):
- Proyek berhasil menghasilkan model *machine learning* untuk memberikan rekomendasi buku yang dipersonalisasi bagi pengguna menggunakan teknik *content-based filtering*.
- Proyek juga berhasil membangun model rekomendasi buku berdasarkan data *rating* dengan teknik *collaborative filtering* yang memungkinkan komunitas untuk merekomendasikan buku lain yang mungkin disukai dan belum pernah dibaca oleh pengguna.
- Melalui analisis data pengguna dan buku dari dataset *Book-Crossing*, proyek ini berhasil memberikan rekomendasi buku yang relevan dan bermanfaat bagi penerbit, penjual buku, serta pengguna dalam menemukan buku yang sesuai dengan minat mereka.

## Referensi

[[1] Mathew, P., Kuriakose, B., & Hegde, V. (2016, March). Book Recommendation System through content based and collaborative filtering method. In *2016 International conference on data mining and advanced computing (SAPIENCE)* (pp. 47-52). IEEE.](https://ieeexplore.ieee.org/abstract/document/7684166)

[[2] Ng, Y. K. (2020). CBRec: a book recommendation system for children using the matrix factorisation and content-based filtering approaches. *International Journal of Business Intelligence and Data Mining, 16*(2), 129-149.](https://www.inderscienceonline.com/doi/abs/10.1504/IJBIDM.2020.104738)
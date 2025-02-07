# Laporan Proyek Machine Learning - Okta Agnes Ladyagatha Manik

## Domain Proyek

Pertanian adalah sektor yang sangat penting bagi perekonomian dan ketahanan pangan suatu negara. Di Indonesia, mayoritas masyarakat bergantung pada sektor pertanian untuk memenuhi kebutuhan hidup sehari-hari. Namun, petani sering kali menghadapi tantangan dalam memilih jenis tanaman yang paling sesuai dengan kondisi lahan yang mereka miliki. Pemilihan yang tidak tepat dapat mengakibatkan kerugian besar, mulai dari rendahnya hasil panen hingga kerusakan lingkungan.

Masalah utama yang dihadapi adalah kurangnya informasi mengenai kecocokan jenis tanaman dengan kondisi lahan yang ada. Tanah yang berbeda-beda memiliki sifat fisik dan kimia yang berbeda pula, seperti pH, tekstur, kelembapan, dan kadar hara, yang semuanya mempengaruhi pertumbuhan tanaman. Selain itu, faktor iklim seperti suhu, curah hujan, dan kelembapan udara juga memainkan peran penting dalam menentukan tanaman apa yang dapat tumbuh dengan optimal di suatu lokasi.

Pentingnya memilih tanaman yang tepat berdasarkan kondisi lahan memotivasi pengembangan sistem berbasis data yang dapat memberikan rekomendasi otomatis mengenai tanaman yang paling cocok untuk lahan tertentu. Penggunaan teknologi informasi dan sistem berbasis data dapat membantu petani dalam membuat keputusan yang lebih baik[1]. Sistem ini menggunakan data historis mengenai kondisi tanah dan iklim untuk menghasilkan prediksi yang lebih akurat mengenai jenis tanaman yang dapat tumbuh dengan baik pada lahan tersebut.

Selain itu, riset menunjukkan bahwa teknologi seperti data mining dan machine learning dapat digunakan untuk menganalisis data besar yang berkaitan dengan karakteristik tanah dan pola iklim[2]. Dengan pendekatan ini, sistem dapat memberikan rekomendasi yang lebih personalisasi, bahkan untuk kondisi lahan yang lebih spesifik. Hal ini tentu sangat menguntungkan bagi petani, karena mereka dapat mendapatkan hasil yang lebih optimal dengan memilih jenis tanaman yang sesuai.

Sistem prediksi ini bukan hanya berguna untuk meningkatkan hasil pertanian, tetapi juga untuk mendukung keberlanjutan pertanian dengan mengurangi risiko kerusakan lingkungan akibat kesalahan dalam memilih tanaman. Penggunaan sistem berbasis data dapat meningkatkan efisiensi penggunaan sumber daya alam seperti air dan pupuk, yang tentunya berkontribusi pada praktik pertanian yang lebih berkelanjutan[3]. Pemilihan jenis tanaman yang tepat sangat penting untuk keberhasilan pertanian. Tanpa adanya pendekatan yang berbasis data, petani sering kali melakukan percobaan yang berisiko, yang dapat mengarah pada kerugian ekonomi dan kerusakan lingkungan. Untuk itu, solusi berbasis data yang dapat dapat meprediksi mengenai jenis tanaman yang sesuai untuk suatu lahan sangat diperlukan.

## Business Understanding

Penggunaan sistem prediksi berbasis data dan teknologi informasi dapat membantu menyelesaikan masalah ini dengan memberikan solusi yang lebih akurat dan efisien. Dengan memanfaatkan algoritma dan model berbasis data, kita dapat menganalisis variabel-variabel penting seperti kondisi tanah dan kelembapan udar untuk memprediksi tanaman yang paling cocok untuk lahan tertentu.

### Problem Statements

- Pernyataan Masalah 1: Petani sering kali kesulitan dalam memilih jenis tanaman yang sesuai dengan kondisi lahan mereka. Keputusan yang diambil biasanya tidak berbasis pada analisis data yang objektif dan hanya mengandalkan pengalaman atau kebiasaan lokal. Hal ini dapat menyebabkan pemilihan tanaman yang tidak cocok dengan kondisi tanah dan kelembapan udara, yang pada gilirannya dapat mengurangi hasil pertanian.
- Pernyataan Masalah 2: Keterbatasan pengetahuan mengenai kebutuhan nutrisi tanah dan pengaruh faktor iklim seperti suhu, kelembapan, dan curah hujan terhadap pertumbuhan tanaman menyebabkan hasil pertanian tidak optimal. Misalnya, tingkat nitrogen, fosfor, kalium (N, P, K), pH tanah, suhu, kelembapan, dan curah hujan semuanya mempengaruhi jenis tanaman yang dapat tumbuh dengan baik, namun banyak petani yang tidak mengetahui bagaimana cara mengoptimalkan penggunaan faktor-faktor ini.
- Pernyataan Masalah 3: Perubahan iklim global dapat menyebabkan ketidakpastian terkait pola cuaca dan curah hujan yang mempengaruhi pertumbuhan tanaman. Petani membutuhkan alat bantu untuk memilih jenis tanaman yang dapat beradaptasi dengan fluktuasi iklim yang terjadi.

### Goals

- Jawaban pernyataan masalah 1: Membangun sistem berbasis data yang dapat memberikan rekomendasi jenis tanaman berdasarkan karakteristik tanah dan kondisi iklim yang ada, seperti kandungan N, P, K, suhu, kelembapan, pH tanah, dan curah hujan. Sistem ini akan memberikan rekomendasi yang lebih objektif dan dapat diandalkan daripada mengandalkan pengalaman atau tebakan.
- Jawaban pernyataan masalah 2: Menyediakan sistem prediksi yang dapat menganalisis data terkait N, P, K, suhu, kelembapan, pH, dan curah hujan untuk memberikan rekomendasi jenis tanaman yang cocok untuk lahan tertentu, sehingga para petani dapat meningkatkan hasil pertanian mereka.
- Jawaban pernyataan masalah 3: Menciptakan model yang dapat beradaptasi dengan perubahan iklim dan memberikan rekomendasi tanaman yang tetap sesuai meskipun ada perubahan dalam suhu dan curah hujan, berdasarkan analisis data historis yang ada.

## Data Understanding

Dataset yang digunakan dalam proyek ini berisi informasi historis tentang kondisi lahan pertanian, termasuk fitur-fitur seperti kandungan N (Nitrogen), P (Phosphorus), K (Kalium), suhu, kelembapan, pH tanah, dan curah hujan. Data ini akan digunakan untuk melatih model machine learning yang bertujuan memprediksi jenis tanaman yang paling cocok untuk ditanam di lahan tertentu serta menganalisis perubahan kondisi lahan untuk memberikan rekomendasi tanaman yang optimal di masa depan.<br>
Land Dataset: (https://www.kaggle.com/datasets/siddharthss/crop-recommendation-dataset).

### Variabel-variabel pada land dataset adalah sebagai berikut:

- N: rasio kandungan Nitrogen dalam tanah
- P: rasio kandungan Fosfor dalam tanah
- K: rasio kandungan Kalsium dalam tanah
- temperature: suhu dalam derajat celcius
- humidity: kelembaban relatif dalam %
- pH: nilai pH tanah
- rainfall: curah hujan dalam mm
- label: jenis tanaman yang cocok untuk ditanam di lahan pertanian berdasarkan variabel

Dataset yang mentah tersebut memiliki data yang berjumlah 2200 dan terdiri dari 8 atribut, yaitu N, P, K, temperature, humidity pH, rainfall dan label. Hal tersebut membuat apabila data-data tersebut disusun dalam format tabel baris dan kolom akan membentuk tabel yang berisi oleh 29580 baris dan 5 kolom.

Untuk memahami atribut-atribut yang ada di dalam dataset tersebut dilakukan beberapa langkah untuk memahami isi dan tipe atribut tersebut. Pertama, dengan menggunakan fungsi bawaan dari python yaitu .info() penulis bisa mendapatkan bahwa dalam dataset tersebut tidak terdapat data yang kosong dan bisa mengetahui tipe data dari masing-masing atribut yang ada pada dataset.<br>
![alt text](https://github.com/oktaagnes/MLT_prediksi_Laporan-Proyek_Machine_Learning/blob/main/assets/1.png?raw=true)<br>
gambar 1. keluaran dari built-in function bahasa pemrograman Python pada dataset land_df<br>

Kedua, dengan menggunakan .describe() penulis dapat mengetahui statistik dasar dari data seperti percentile, mean, standar deviasi, jumlah data, min, dan max. Hasil fungsi ini ditampilkan pada tabel 2 seperti berikut.
![alt text](https://github.com/oktaagnes/MLT_prediksi_Laporan-Proyek_Machine_Learning/blob/main/assets/2.png?raw=true)<br>
gambar 2. keluaran dari statistik dataset land_df hasil dari fungsi .describe()<br>

Pada notebooks dilakukan visualisasi untuk membandingkan rerata data keseluruhan yang bertipe house dan yang bertipe unit. Didapatkan bahwa rerata harga rumah yang bertipe house lebih tinggi daripada rerata harga rumah yang bertipe unit.

Pada atribut fitur price dilakukan visualisasi data seperti tampak di gambar 3, visualisasi harga yang dilakukan adalah dengan memanfaatkan histogram untuk mengetahui jumlah data pada masing-masing rentang harga rumah yang ada didataset.<br>
![alt text](https://github.com/oktaagnes/MLT_prediksi_Laporan-Proyek_Machine_Learning/blob/main/assets/sebaranpng.png?raw=true)<br>
gambar 3. visual sebaran data numerikal <br>

Pada atribut fitur price dilakukan visualisasi data seperti tampak di gambar 3, visualisasi harga yang dilakukan adalah dengan memanfaatkan histogram untuk mengetahui jumlah data pada masing-masing rentang harga rumah yang ada didataset.<br>
![alt text](https://github.com/oktaagnes/MLT_prediksi_Laporan-Proyek_Machine_Learning/blob/main/assets/sebaran%20label.png?raw=true)<br>
gambar 4. visual sebaran data katrgorikal <br>

Selanjutnya untuk mengetahui hubungan masing-masing fitur terhadap satu sama lain dihitung korelasinya. Menghasilkan bahwa fitur price memiliki korelasi positif terhadap fitur jumlah bedrooms. Karena terdapat sejumlah data yang tidak konsisten, dalam arti ada data yang jumlah bedrooms yang tinggi tetapi memiliki price yang tinggi dan ada juga rendah. Dengan begitu nilai korelasi tidak mendekati positif satu, hanya bernilai 0.48.<br>
![alt text](https://github.com/oktaagnes/MLT_prediksi_Laporan-Proyek_Machine_Learning/blob/main/assets/matrix.png?raw=true) <br>
gambar 5. matriks <br>

## Data Preparation

Dalam data preparation, 6 hal yang akan dilakukan sebelum memasukkan data ke model latih:

<li>Penghapusan data duplikat, Penghapusan data duplikat penting agar analisis data lebih akurat dan tidak ada perhitungan ganda. keluaran saat Pengecekan data frame yaitu, false. Artinya tidak ada data duplikat pada data frame, sehingga tidak perlu penghapusan data duuplikat</li>
<li>Penanganan outlier dengan metode IQR. Outlier adalah data yang terlalu kecil atau terlalu besar dibandingkan data lainnya, sehingga bisa mengganggu analisis. IQR (Interquartile Range) adalah metode statistik yang digunakan untuk menentukan batas normal sebuah data dan mengidentifikasi outlier.
Q1 (Kuartil 1) = nilai 25% data terendah
Q3 (Kuartil 3) = nilai 75% data terendah <br>
Batas bawah = Q1 - (1.5 × IQR) <br>
Batas atas = Q3 + (1.5 × IQR).</li> <br>
<p><strong>Rumus IQR:</strong></p>
    <p>$$ IQR = Q3 - Q1 $$</p>
<li> Principal Component Analysis (PCA), sebuah teknik yang digunakan untuk mengubah data dengan banyak variabel (fitur) menjadi representasi yang lebih sederhana, sambil tetap mempertahankan sebagian besar informasi yang terkandung dalam data asli. PCA digunakan untuk mengurangi kompleksitas data, menghilangkan korelasi antar variabel, serta meningkatkan interpretabilitas tanpa mengorbankan informasi penting dalam data.</li>
<li>Encoding Fitur Kategorik : Encoding fitur kategorik dilaksanakan di beberapa fitur yang bertipe object. Hal ini dilakukan karena model machine learning hanya dapat menerima data dalam bentuk numerik. Untuk encoding fitur menggunakan LabelEncoder.</li>
<li>Pembagian dataset, penulis memisahkan variabel independen (N, P, K, suhu, kelembaban, pH, curah hujan) sebagai data.
𝑋 
X, dan variabel dependen (label) sebagai data.
𝑦.<br> Membagi dataset menjadi dua bagian, yaitu training set dan test set. Pembagian ini penting untuk menghindari risiko overfitting, yang terjadi ketika model "menghafal" data latih dan tidak dapat generalisasi dengan baik pada data yang belum pernah dilihat sebelumnya. Dengan membagi dataset, model dapat diuji untuk memastikan kemampuannya dalam memprediksi data yang baru dan tidak terduga.<br> </li>

<li> Mendeteksi outlier dalam dataset menggunakan metode berbasis density . Selain itu, proses standarisasi data dilakukan untuk memastikan bahwa seluruh nilai dari fitur numerik, baik pada data latih maupun data uji, berada dalam skala yang seragam, sehingga model dapat memproses data dengan lebih baik dan menghasilkan prediksi yang akurat. </li>

## Modeling

Metode yang penulis pilih untuk memprediksi jenis tanaman yang paling sesuai dengan kondisi lahan, proyek ini menggunakan tiga algoritma machine learning yang kuat dan populer, yaitu K-Nearest Neighbor (KNN), Random Forest, dan XGBoost. Ketiga algoritma ini dipilih karena kemampuannya dalam menangani data numerik yang kompleks dan memberikan hasil yang akurat dalam klasifikasi dan regresi.

- 1. K-Nearest Neighbor (KNN): KNN adalah algoritma berbasis instance yang digunakan untuk menemukan kecocokan antara kondisi lahan dan tanaman yang ada dalam dataset. Pada KNN, setiap titik data (dalam hal ini, kondisi lahan) dibandingkan dengan titik data lainnya, dan tanaman yang paling sering muncul di antara titik data terdekat akan dipilih sebagai rekomendasi. Kelebihan KNN adalah kesederhanaannya dan kemampuannya untuk memberikan prediksi berdasarkan data yang ada tanpa perlu melatih model secara intensif. KNN sangat efektif dalam menangani data dengan hubungan yang tidak linier dan memerlukan sedikit asumsi tentang distribusi data.
     Metode K-Nearest Neighbors (KNN) digunakan untuk melakukan regresi dan klasifikasi dengan berbagai parameter yang disesuaikan guna mendapatkan performa terbaik. Parameter utama yang digunakan dalam model regresi KNeighborsRegressor adalah n_neighbors, yang menentukan jumlah tetangga terdekat dalam proses prediksi. Untuk menentukan nilai optimal dari parameter ini, dilakukan pengujian dengan variasi nilai n_neighbors dari 1 hingga 20.
     Selain itu, untuk model klasifikasi, digunakan KNeighborsClassifier dengan parameter utama yang sama, yaitu n_neighbors, yang dalam penelitian ini ditetapkan sebesar 3. Model klasifikasi dievaluasi menggunakan classification report, yang mencakup metrik precision, recall, dan F1-score, guna menilai kinerja model dalam mengklasifikasikan data dengan lebih akurat.<br>
     ![alt text](https://github.com/oktaagnes/MLT_prediksi_Laporan-Proyek_Machine_Learning/blob/main/assets/cnn.png) <br>
     gambar 7. algoritma KNN <br>

- 2. Random Forest: Random Forest adalah algoritma ensemble yang terdiri dari banyak pohon keputusan (decision trees) yang bekerja bersama-sama untuk meningkatkan akurasi prediksi. Setiap pohon keputusan dilatih menggunakan subset acak dari data, dan hasil akhir dihitung berdasarkan prediksi mayoritas dari seluruh pohon. Dalam konteks prediksi tanaman, Random Forest dapat menangani berbagai faktor tanah dan iklim dengan cara yang lebih fleksibel dan mengurangi risiko overfitting. Algoritma ini sangat kuat dalam menangani variabel yang memiliki interaksi kompleks, seperti suhu, kelembapan, pH, dan unsur hara tanah (N, P, K). Terdapat beberapa parameter yang digunakan untuk menginisialisasi model Random Forest, baik untuk classification maupun regression. Pada RandomForestClassifier, parameter yang digunakan antara lain adalah n_estimators yang bernilai 100, yang menentukan jumlah pohon keputusan (decision trees) yang akan dibangun dalam model.<br>
     ![alt text](<https://github.com/oktaagnes/MLT_prediksi_Laporan-Proyek_Machine_Learning/blob/main/assets/cnn.png?raw=true()>) <br>
     gambar 8. algoritma Random Forest (rf) <br>

- 3. XGBoost: XGBoost (Extreme Gradient Boosting) adalah algoritma yang berbasis pada teknik gradient boosting dan sangat populer karena kemampuannya untuk memberikan akurasi tinggi dengan waktu komputasi yang efisien. XGBoost membangun model secara bertahap, dengan setiap model baru memperbaiki kesalahan model sebelumnya. Dalam prediksi pemilihan tanaman, XGBoost sangat baik untuk menangani data dengan banyak fitur dan menghasilkan rekomendasi yang lebih tepat dengan menggunakan teknik regularisasi untuk mengurangi overfitting. Terdapat beberapa parameter yang digunakan untuk menginisialisasi model XGBoost dan AdaBoost, baik untuk klasifikasi maupun regresi. Pada XGBClassifier, parameter objective diatur dengan nilai multi:softmax, yang menunjukkan bahwa model ini digunakan untuk masalah klasifikasi multikelas, di mana setiap sampel hanya dapat memiliki satu label kelas. Selain itu, parameter num_class ditetapkan sebesar 3, yang menunjukkan bahwa model ini akan mengklasifikasikan data ke dalam tiga kelas yang berbeda.
     Sementara itu, pada model AdaBoostRegressor, terdapat dua parameter utama yang digunakan. Pertama, n_estimators yang bernilai 90, yang menentukan jumlah model dasar (misalnya pohon keputusan) yang akan digunakan dalam proses boosting. Semakin banyak model dasar, model dapat menjadi lebih kuat, tetapi dengan risiko overfitting jika jumlahnya terlalu banyak.<br>
     ![alt text](https://github.com/oktaagnes/MLT_prediksi_Laporan-Proyek_Machine_Learning/blob/main/assets/XGBoost.png?raw=true)<br>
     gambar 9. algoritma XGBoost <br>

## Evaluation
Evaluasi menggunakan Accuracy, Precision, dan Recall bertujuan untuk memberikan gambaran yang lebih komprehensif tentang performa model dalam menangani data, khususnya dalam konteks klasifikasi. Masing-masing metrik ini memberikan informasi yang berbeda mengenai kualitas prediksi model. Berikut adalah penjelasan tentang kegunaan masing-masing metrik:<br>

- Accuracy memberikan gambaran umum performa model.<br>
  ![alt text](https://github.com/oktaagnes/MLT_prediksi_Laporan-Proyek_Machine_Learning/blob/main/assets/formula%20accuracy.png?raw=true)<br>
  gambar 11. formula accuracy <br>
- Precision adalah metrik evaluasi yang mengukur seberapa tepat model Anda dalam membuat prediksi positif. Artinya, dari semua prediksi positif yang dibuat oleh model, berapa banyak yang benar-benar positif.<br>
  ![alt text](https://github.com/oktaagnes/MLT_prediksi_Laporan-Proyek_Machine_Learning/blob/main/assets/formula%20precision.png?raw=true)
  gambar 12. formula precision <br>
- Recall, yang merupakan harmoni antara precision dan recall, dapat digunakan jika Anda ingin menggabungkan keduanya menjadi satu metrik untuk evaluasi yang lebih menyeluruh.<br>
  ![alt text](https://github.com/oktaagnes/MLT_prediksi_Laporan-Proyek_Machine_Learning/blob/main/assets/formula%20recall.png?raw=true)<br>
  gambar 13. formula Recall<br>
  Berikut hasil Evaluasi dari algoritma K-Nearest Neighbor (KNN), Random Forest dan XGBoost. Menggunakan Accuray, precision dan recall.
  ![alt text](https://github.com/oktaagnes/MLT_prediksi_Laporan-Proyek_Machine_Learning/blob/main/assets/akurasi.png)<br>
  gambar 14. hasil evaluasi<br>
  
Metrik evaluasi yang digunakan pada proyek ini adalah mean squared error (MSE) dan menggunakan Accuracy (%), Precision (%), dan Recall (%). Mean squared error (MSE) mengukur error dalam model statistik dengan cara menghitung rata-rata error dari kuadrat hasil aktual dikurang hasil prediksi. Berikut merupakan formula MSE: <br/>
$$\text{MSE}(y, \hat{y}) = \frac{\sum_{i=0}^{N - 1} (y_i - \hat{y}_i)^2}{N}$$ <br>
![alt text](https://github.com/oktaagnes/MLT_prediksi_Laporan-Proyek_Machine_Learning/blob/main/assets/mse.png)<br>
gambar 10. Mean squared error (MSE) <br>
Berdasarkan hasil evaluasi model di atas, dapat disimpulkan bahwa model terbaik untuk melakukan prediksi Pemilihan Jenis Tanaman untuk Lahan Pertanian adalah model Random Forest dapat dianggap sebagai model terbaik untuk digunakan dalam kasus ini karena memiliki <b> accuracy 99%<b>.
Model yang diuji berhasil menjawab setiap problem statement dengan baik, terutama dengan performa tinggi dari Random Forest. Model ini dapat digunakan sebagai alat bantu bagi petani untuk memilih jenis tanaman berdasarkan data objektif, mengoptimalkan hasil pertanian, dan beradaptasi dengan perubahan iklim.

## Referensi

- [1] http://eprints.bsi.ac.id/index.php/co-science/article/view/2987/1686 <br>
- [2] https://jurnal.itscience.org/index.php/digitech/article/view/2852/2200 <br>
- [3] https://prosiding.stis.ac.id/index.php/semnasoffstat/article/view/230/22

# Laporan Proyek Machine Learning - Okta Agnes Ladyagatha Manik

## Domain Proyek

Pertanian adalah sektor yang sangat penting bagi perekonomian dan ketahanan pangan suatu negara. Di Indonesia, mayoritas masyarakat bergantung pada sektor pertanian untuk memenuhi kebutuhan hidup sehari-hari. Namun, petani sering kali menghadapi tantangan dalam memilih jenis tanaman yang paling sesuai dengan kondisi lahan yang mereka miliki. Pemilihan yang tidak tepat dapat mengakibatkan kerugian besar, mulai dari rendahnya hasil panen hingga kerusakan lingkungan.

Masalah utama yang dihadapi adalah kurangnya informasi mengenai kecocokan jenis tanaman dengan kondisi lahan yang ada. Tanah yang berbeda-beda memiliki sifat fisik dan kimia yang berbeda pula, seperti pH, tekstur, kelembapan, dan kadar hara, yang semuanya mempengaruhi pertumbuhan tanaman. Selain itu, faktor iklim seperti suhu, curah hujan, dan kelembapan udara juga memainkan peran penting dalam menentukan tanaman apa yang dapat tumbuh dengan optimal di suatu lokasi.

Pentingnya memilih tanaman yang tepat berdasarkan kondisi lahan memotivasi pengembangan sistem berbasis data yang dapat memberikan rekomendasi otomatis mengenai tanaman yang paling cocok untuk lahan tertentu. Penggunaan teknologi informasi dan sistem berbasis data dapat membantu petani dalam membuat keputusan yang lebih baik[1]. Sistem ini menggunakan data historis mengenai kondisi tanah dan iklim untuk menghasilkan prediksi yang lebih akurat mengenai jenis tanaman yang dapat tumbuh dengan baik pada lahan tersebut.

Selain itu, riset menunjukkan bahwa teknologi seperti data mining dan machine learning dapat digunakan untuk menganalisis data besar yang berkaitan dengan karakteristik tanah dan pola iklim[2]. Dengan pendekatan ini, sistem dapat memberikan rekomendasi yang lebih personalisasi, bahkan untuk kondisi lahan yang lebih spesifik. Hal ini tentu sangat menguntungkan bagi petani, karena mereka dapat mendapatkan hasil yang lebih optimal dengan memilih jenis tanaman yang sesuai.

Sistem prediksi ini bukan hanya berguna untuk meningkatkan hasil pertanian, tetapi juga untuk mendukung keberlanjutan pertanian dengan mengurangi risiko kerusakan lingkungan akibat kesalahan dalam memilih tanaman. Penggunaan sistem berbasis data dapat meningkatkan efisiensi penggunaan sumber daya alam seperti air dan pupuk, yang tentunya berkontribusi pada praktik pertanian yang lebih berkelanjutan[3]. Pemilihan jenis tanaman yang tepat sangat penting untuk keberhasilan pertanian. Tanpa adanya pendekatan yang berbasis data, petani sering kali melakukan percobaan yang berisiko, yang dapat mengarah pada kerugian ekonomi dan kerusakan lingkungan. Untuk itu, solusi berbasis data yang dapat dapat meprediksi mengenai jenis tanaman yang sesuai untuk suatu lahan sangat diperlukan.

Penggunaan sistem prediksi berbasis data dan teknologi informasi dapat membantu menyelesaikan masalah ini dengan memberikan solusi yang lebih akurat dan efisien. Dengan memanfaatkan algoritma dan model berbasis data, kita dapat menganalisis variabel-variabel penting seperti kondisi tanah dan kelembapan udar untuk memprediksi tanaman yang paling cocok untuk lahan tertentu.

## Business Understanding

Pada bagian ini, kita akan membahas klarifikasi masalah yang terkait dengan proyek prediksi pemilihan jenis tanaman untuk lahan pertanian tertentu dengan memanfaatkan data yang ada. Bagian laporan ini mencakup:

### Problem Statements
- Pernyataan Masalah 1: Petani sering kali kesulitan dalam memilih jenis tanaman yang sesuai dengan kondisi lahan mereka. Keputusan yang diambil biasanya tidak berbasis pada analisis data yang objektif dan hanya mengandalkan pengalaman atau kebiasaan lokal. Hal ini dapat menyebabkan pemilihan tanaman yang tidak cocok dengan kondisi tanah dan kelembapan udara, yang pada gilirannya dapat mengurangi hasil pertanian.
- Pernyataan Masalah 2: Keterbatasan pengetahuan mengenai kebutuhan nutrisi tanah dan pengaruh faktor iklim seperti suhu, kelembapan, dan curah hujan terhadap pertumbuhan tanaman menyebabkan hasil pertanian tidak optimal. Misalnya, tingkat nitrogen, fosfor, kalium (N, P, K), pH tanah, suhu, kelembapan, dan curah hujan semuanya mempengaruhi jenis tanaman yang dapat tumbuh dengan baik, namun banyak petani yang tidak mengetahui bagaimana cara mengoptimalkan penggunaan faktor-faktor ini.
- Pernyataan Masalah 3: Perubahan iklim global dapat menyebabkan ketidakpastian terkait pola cuaca dan curah hujan yang mempengaruhi pertumbuhan tanaman. Petani membutuhkan alat bantu untuk memilih jenis tanaman yang dapat beradaptasi dengan fluktuasi iklim yang terjadi.

### Goals
- Jawaban pernyataan masalah 1: Membangun sistem berbasis data yang dapat memberikan rekomendasi jenis tanaman berdasarkan karakteristik tanah dan kondisi iklim yang ada, seperti kandungan N, P, K, suhu, kelembapan, pH tanah, dan curah hujan. Sistem ini akan memberikan rekomendasi yang lebih objektif dan dapat diandalkan daripada mengandalkan pengalaman atau tebakan.
- Jawaban pernyataan masalah 2: Menyediakan sistem prediksi yang dapat menganalisis data terkait N, P, K, suhu, kelembapan, pH, dan curah hujan untuk memberikan rekomendasi jenis tanaman yang cocok untuk lahan tertentu, sehingga para petani dapat meningkatkan hasil pertanian mereka.
- Jawaban pernyataan masalah 3: Menciptakan model yang dapat beradaptasi dengan perubahan iklim dan memberikan rekomendasi tanaman yang tetap sesuai meskipun ada perubahan dalam suhu dan curah hujan, berdasarkan analisis data historis yang ada.

## Data Understanding
Dataset yang digunakan dalam proyek ini berisi informasi historis tentang kondisi lahan pertanian, termasuk fitur-fitur seperti kandungan N (Nitrogen), P (Phosphorus), K (Kalium), suhu, kelembapan, pH tanah, dan curah hujan. Data ini akan digunakan untuk melatih model machine learning yang bertujuan memprediksi jenis tanaman yang paling cocok untuk ditanam di lahan tertentu serta menganalisis perubahan kondisi lahan untuk memberikan rekomendasi tanaman yang optimal di masa depan. 
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

Untuk memahami atribut-atribut yang ada di dalam dataset tersebut dilakukan beberapa langkah untuk memahami isi dan tipe atribut tersebut. Pertama, dengan menggunakan fungsi bawaan dari python yaitu .info() penulis bisa mendapatkan bahwa dalam dataset tersebut tidak terdapat data yang kosong dan bisa mengetahui tipe data dari masing-masing atribut yang ada pada dataset.

Tabel 1. keluaran dari built-in function bahasa pemrograman Python pada dataset land_df
(letak gambar tabel 1)
Kedua, dengan menggunakan .describe() penulis dapat mengetahui statistik dasar dari data seperti percentile, mean, standar deviasi, jumlah data, min, dan max. Hasil fungsi ini ditampilkan pada tabel 2 seperti berikut.

Tabel 2. statistik dataset land_df hasil dari fungsi .describe()
(letak gambar tabel 2)
Pada notebooks dilakukan visualisasi untuk membandingkan rerata data keseluruhan yang bertipe house dan yang bertipe unit. Didapatkan bahwa rerata harga rumah yang bertipe house lebih tinggi daripada rerata harga rumah yang bertipe unit.



**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
## Referensi
[1] http://eprints.bsi.ac.id/index.php/co-science/article/view/2987/1686
[2] https://jurnal.itscience.org/index.php/digitech/article/view/2852/2200
[3] https://prosiding.stis.ac.id/index.php/semnasoffstat/article/view/230/22
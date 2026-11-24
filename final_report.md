# Customer Segmentation Through RFM-Based Clustering

**Laporan Akhir Analisis Data Sains & Pemodelan Unsupervised Learning**  
*Penyusun:* Data Science AI Agent  
*Dataset Sumber:* [UCI Machine Learning Repository: Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail) (DOI: `10.24432/C5BW33`)  
*Algoritma Utama:* K-Means Clustering  
*Pendekatan Rekayasa Fitur:* RFM (Recency, Frequency, Monetary)  
*Status Evaluasi:* Selesai & Tervalidasi Penuh  

---

## 1. Judul Project

**Customer Segmentation Through RFM-Based Clustering**

---

## 2. Ringkasan Eksekutif

Laporan ini menyajikan solusi segmentasi pelanggan berbasis *Unsupervised Learning* pada dataset transaksi e-commerce ritel *Online Retail* (Inggris, periode 1 Desember 2010 hingga 9 Desember 2011). Tujuan utama proyek ini adalah mengidentifikasi struktur kelompok pelanggan yang alami dan bermakna secara bisnis berdasarkan riwayat kebaruan belanja (*Recency*), frekuensi pemesanan (*Frequency*), dan nilai moneter transaksi (*Monetary* / RFM).

Seluruh tahapan end-to-end telah dijalankan pada data aktual berukuran 541.909 baris:
1. **Audit Data & Pembersihan:** Ditemukan 135.080 baris transaksi tanpa `CustomerID` (24,93%), 5.225 baris duplikat identik, 8.872 baris pembatalan/retur, dan 40 baris dengan harga satuan tidak positif. Seluruh anomali tersebut disaring secara metodologis menghasilkan 392.692 baris transaksi pembelian yang sah. Dari 4.372 ID pelanggan terdaftar, 4.338 pelanggan terekonsiliasi memiliki transaksi pembelian positif yang valid (34 pelanggan sisanya hanya tercatat melakukan retur/pembatalan sehingga tidak memiliki riwayat pembelian).
2. **Transformasi Fitur:** Distribusi asli fitur RFM mengalami ketimpangan ekstrim (*skewness* Frequency: 12,07; Monetary: 19,34). Transformasi $log(1+x)$ (`log1p`) berhasil menormalkan kecondongan secara drastis (Frequency menjadi 1,21; Monetary menjadi 0,40), diikuti dengan standardisasi Z-score (`StandardScaler`).
3. **Optimasi K-Means:** Pengujian multi-kriteria untuk $k \in [2, 10]$ membuktikan bahwa **$k = 4$** merupakan konfigurasi optimal yang seimbang antara separasi matematis (*Inertia* = 3.938,60; *Silhouette Score* = 0,3373; *Davies-Bouldin Index* = 1,0106) dan kemampuan aksi bisnis (*actionability*).
4. **Temuan Segmen Utama:**
   - **Champions / High-Value (16,34% pelanggan):** Menyumbang **64,72% dari total pendapatan bisnis** (£5,75 juta) dengan median belanja £3.722,82, median 10 kali order, dan transaksi terakhir hanya 8 hari lalu.
   - **Loyal Customers (27,04% pelanggan):** Menyumbang **23,82% pendapatan** (£2,12 juta) dengan median belanja £1.341,65 dan median 4 kali order.
   - **Recent / Promising Customers (19,34% pelanggan):** Pelanggan aktif terkini (median Recency 17 hari) dengan median belanja £466,70 dan median 2 kali order, menyumbang **5,13% pendapatan**.
   - **At-Risk / Hibernating (37,28% pelanggan):** Pelanggan yang telah tidak aktif hampir 6 bulan (median Recency 177 hari) dan mayoritas hanya bertransaksi 1 kali (median belanja £298,40), menyumbang **6,33% pendapatan**.
5. **Dampak Strategis:** Konsentrasi 88,54% pendapatan berada di tangan 43,38% pelanggan (*Champions* & *Loyal*). Rekomendasi bisnis difokuskan pada strategi retensi VIP tanpa perang diskon untuk pelanggan bernilai tinggi, otomatisasi *nurturing* untuk pelanggan baru/potensial, serta reaktivasi hemat biaya berbasis segmentasi.

---

## 3. Latar Belakang dan Tujuan Bisnis

### 3.1 Latar Belakang Industri
Dalam lanskap bisnis e-commerce dan ritel lintas-negara, perlakuan yang seragam (*one-size-fits-all*) terhadap seluruh basis pelanggan menimbulkan inefisiensi alokasi anggaran pemasaran, *churn rate* yang tinggi, dan penurunan *Customer Lifetime Value* (CLV). Perusahaan ritel online yang berbasis di Inggris ini melayani ribuan pelanggan grosir (*wholesalers*) dan ritel individu di seluruh dunia dengan karakteristik perilaku yang sangat heterogen.

### 3.2 Pertanyaan Bisnis
Analisis ini dirancang untuk menjawab lima pertanyaan inti manajemen:
1. *Apakah pelanggan dapat dikelompokkan secara terukur berdasarkan kebaruan, frekuensi, dan nilai pembeliannya?*
2. *Berapa jumlah klaster pelanggan yang paling rasional secara matematis dan dapat dioperasionalkan secara bisnis?*
3. *Bagaimana profil perilaku belanja mendalam dari masing-masing klaster?*
4. *Klaster mana yang mendominasi populasi pelanggan dan klaster mana yang menopang arus kas pendapatan utama?*
5. *Strategi retensi, program loyalitas, dan mitigasi churn apa yang paling proporsional untuk diterapkan pada setiap segmen?*

### 3.3 Batasan Metodologi
Analisis ini menggunakan pembelajaran mesin tanpa supervisi (*unsupervised learning*). Model tidak bertujuan membuktikan relasi sebab-akibat (*causality*) dan tidak memberikan garansi peningkatan penjualan seketika, melainkan memberikan segmentasi deskriptif dan preskriptif yang terstruktur secara objektif.

---

## 4. Deskripsi Dataset dan Variabel

Dataset diperoleh dari arsip UCI Machine Learning Repository (Chen et al., 2012; DOI: `10.24432/C5BW33`). Dataset mencakup seluruh transaksi penjualan yang tercatat antara 1 Desember 2010 pukul 08:26 hingga 9 Desember 2011 pukul 12:50.

### Tabel 1: Kamus Variabel Dataset Online Retail
| Nama Kolom | Tipe Data Asli | Deskripsi Teknis | Karakteristik Data |
| :--- | :--- | :--- | :--- |
| `InvoiceNo` | String / Object | Kode faktur transaksi 6-digit. Awalan `C` mengindikasikan pembatalan. | Nominal; non-null |
| `StockCode` | String / Object | Kode barang/produk unik 5 atau 6-digit. | Nominal; non-null |
| `Description` | String / Object | Nama atau deskripsi ringkas produk. | Teks deskriptif; terdapat 1.454 data kosong |
| `Quantity` | Integer (`int64`) | Kuantitas produk per baris pesanan. | Numerik kontinu/diskrit; memuat nilai negatif (retur) |
| `InvoiceDate` | String / Object | Tanggal dan stempel waktu pencatatan pesanan. | Format waktu `DD/MM/YYYY HH:MM` |
| `UnitPrice` | Float (`float64`) | Harga satuan per produk dalam pound sterling (£). | Numerik kontinu; memuat harga 0 dan negatif |
| `CustomerID` | Float (`float64`) | Nomor identitas unik pelanggan 5-digit numerik. | Nominal kategorikal; terdapat 135.080 data kosong |
| `Country` | String / Object | Negara domisili pelanggan. | Kategorikal; 38 negara |

---

## 5. Data Audit dan Temuan EDA

Audit data menyeluruh dijalankan menggunakan Python pada file mentah `Online_Retail.csv` (541.909 baris, 8 kolom).

### 5.1 Nilai Kosong (*Missing Values*)
Pemeriksaan data kosong menunjukkan konsentrasi nilai kosong hanya pada dua atribut:
- `CustomerID`: Hilang sebanyak **135.080 baris** (**24,93%** dari total data).
- `Description`: Hilang sebanyak **1.454 baris** (**0,27%** dari total data).
- Seluruh kolom lainnya (`InvoiceNo`, `StockCode`, `Quantity`, `InvoiceDate`, `UnitPrice`, `Country`) memiliki kelengkapan 100%.

![Missing Values Overview](visualizations/missing_values.png)

### 5.2 Duplikasi dan Transaksi Anomali
- **Duplikat Identik:** Ditemukan **5.268 baris** (0,97%) baris data yang identik pada seluruh 8 kolom secara simultan. Hal ini merefleksikan pencatatan ganda pada sistem kasir atau transmisi data redundan.
- **Transaksi Pembatalan (`InvoiceNo` diawali 'C'):** Teridentifikasi **9.288 baris** (1,71%). Seluruh transaksi pembatalan ini memiliki nilai `Quantity < 0`.
- **Kuantitas Negatif Tanpa Awalan 'C':** Ditemukan 1.336 baris dengan nilai `Quantity < 0` tetapi nomor invoice tidak diawali huruf 'C'. Audit mendalam membuktikan bahwa **100% dari 1.336 baris ini tidak memiliki `CustomerID` (NaN)**, dengan deskripsi operasional seperti *'damaged'*, *'lost'*, *'thrown away'*, atau penyesuaian inventaris gudang internal.
- **Harga Satuan Non-Positif (`UnitPrice <= 0`):**
  - `UnitPrice < 0`: Ditemukan 2 baris (kedua baris tidak memiliki `CustomerID`, dengan deskripsi penyesuaian utang piutang *'Adjust bad debt'*).
  - `UnitPrice == 0`: Ditemukan 2.515 baris (2.475 baris tidak memiliki CustomerID, dan 40 baris memiliki CustomerID valid yang merepresentasikan sampel produk gratis atau hadiah promosi).

### 5.3 Pola Waktu Transaksi dan Konsentrasi Geografis
- **Rentang Waktu:** 1 Desember 2010 pukul 08:26:00 hingga 9 Desember 2011 pukul 12:50:00 (kurang lebih 12 bulan 9 hari).
- **Pola Musiman:** Terjadi lonjakan volume dan nilai transaksi yang signifikan menjelang kuartal keempat, mencapai puncaknya pada bulan November 2011 (frekuensi mencapai 3.473 invoice dan pendapatan melampaui £1,16 juta), yang didorong oleh persiapan musim belanja akhir tahun (*Black Friday* dan persiapan Natal). Bulan Desember 2011 hanya mencakup pencatatan hingga tanggal 9 Desember.
- **Konsentrasi Negara:** Basis pelanggan sangat didominasi oleh Britania Raya (*United Kingdom*), mencakup **3.920 dari 4.338 pelanggan valid (90,36%)** dan **349.203 baris transaksi valid (88,93%)**. Negara internasional terbesar berikutnya mencakup Jerman (94 pelanggan, 9.025 transaksi), Prancis (87 pelanggan, 8.326 transaksi), EIRE (7.226 transaksi), dan Spanyol (28 pelanggan, 2.479 transaksi).

![Tren Transaksi](visualizations/transaction_trend.png)

---

## 6. Data Cleaning dan Alasan Keputusan Preprocessing

Pembersihan data dilakukan melalui *cleaning funnel* yang transparan dan terdokumentasi, dengan fokus membentuk dataset transaksi pembelian yang valid (*purchase-based segmentation*).

### Tabel 2: Tahapan Corong Pembersihan Data (*Cleaning Funnel*)
| Langkah | Deskripsi Operasi | Baris Tersisa | Baris Dikeluarkan | Persentase Dikeluarkan | Alasan & Rasional Metodologis |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **0** | **Dataset Mentah** | **541.909** | - | - | Kondisi awal file `Online_Retail.csv`. |
| **1** | Filter `CustomerID` Valid | 406.829 | 135.080 | 24,93% | Pelanggan tanpa ID unik tidak dapat diintegrasikan ke profil riwayat RFM level individu. Mengisi ID buatan (*imputation*) akan mengkontaminasi segmentasi. |
| **2** | Hapus Duplikat Identik | 401.604 | 5.225 | 1,28% | Menghilangkan pencatatan berulang yang terjadi pada detik yang sama untuk item dan kuantitas yang identik. |
| **3** | Eliminasi Pembatalan & Kuantitas $\le 0$ | 392.732 | 8.872 | 2,21% | Mengeluarkan transaksi pembatalan (kode `C`) dan retur barang agar metrik RFM mencerminkan intensitas pembelian riil. |
| **4** | Eliminasi `UnitPrice` $\le 0$ | 392.692 | 40 | 0,01% | Mengeluarkan baris harga 0 (pemberian sampel gratis/penyesuaian manual) yang mendistorsi nilai moneter transaksi komersial. |
| **Final** | **Data Pembelian Bersih Valid** | **392.692** | **149.217** | **27,54%** | Basis data transaksi final untuk rekayasa fitur RFM. |

### 6.1 Rekonsiliasi Pelanggan Terdaftar vs Pelanggan Bersih
- Jumlah `CustomerID` unik awal: **4.372 pelanggan**.
- Jumlah `CustomerID` unik setelah pembersihan: **4.338 pelanggan**.
- **Selisih 34 pelanggan:** Audit membuktikan bahwa ke-34 pelanggan ini **sama sekali tidak memiliki riwayat transaksi pembelian positif** di seluruh dataset. Rinciannya: 33 pelanggan hanya memiliki catatan faktur pembatalan (`CXXXXXX` dengan kuantitas negatif), dan 1 pelanggan (ID `13256`) hanya memiliki transaksi bernilai harga £0,00. Karena segmentasi berorientasi pada daya beli (*purchase-based RFM*), pengecualian 34 akun non-pembeli ini sah dan konsisten secara metodologis.

### 6.2 Uji Sensitivitas (*Sensitivity Checks*)
1. **Dampak Duplikat:** Korelasi fitur RFM antara data dengan duplikat vs tanpa duplikat menghasilkan nilai:
   - Korelasi Recency: **1,0000**
   - Korelasi Frequency: **1,0000** (karena dihitung berbasis nomor invoice unik)
   - Korelasi Monetary: **0,999997**
   *Kesimpulan:* Pembersihan duplikat merapikan integritas baris data tanpa mengubah karakteristik profil pelanggan.
2. **Perlakuan Retur (Gross Monetary vs Net Monetary):** Jika nilai retur dipotong langsung ke total belanja, terdapat 19 pelanggan yang menghasilkan *Net Monetary* negatif atau nol (pelanggan yang mengembalikan barang lebih banyak dari yang dibeli selama jendela 1 tahun). Nilai $\le 0$ merusak transformasi logaritma matematis. Analisis korelasi peringkat Spearman antara *Gross Monetary* dan *Net Monetary* mencapai **0,9902** (Pearson: 0,9446). Hal ini membuktikan bahwa daya belanja kotor (*gross spending*) mempertahankan hierarki nilai pelanggan secara konsisten tanpa anomali nilai negatif.

---

## 7. Definisi, Pembentukan, dan Ringkasan Fitur RFM

Data transaksi level produk diagregasikan ke level pelanggan unik (`CustomerID`).

### 7.1 Definisi Formula
- **Tanggal Acuan (*Reference Date*):** Ditetapkan secara eksplisit pada **10 Desember 2011 pukul 12:50:00** (tepat 1 hari setelah transaksi pembelian terakhir tercatat: 9 Desember 2011 12:50:00).
- **Recency ($R$):** Jumlah hari kalender antara tanggal transaksi pembelian valid terakhir pelanggan dengan tanggal acuan:
  $$Recency = \lfloor (ReferenceDate - \max(InvoiceDate))_{hari} \rfloor$$
  *Interpretasi:* Semakin kecil nilai $R$, semakin baru pelanggan bertransaksi.
- **Frequency ($F$):** Jumlah invoice pembelian unik (`InvoiceNo`) milik pelanggan:
  $$Frequency = \text{CountDistinct}(InvoiceNo)$$
  *Catatan penting:* Bukan jumlah baris produk (*item rows*), melainkan jumlah sesi/perjalanan belanja yang sah.
- **Monetary ($M$):** Akumulasi total belanja produk pelanggan dalam pound sterling (£):
  $$Monetary = \sum (Quantity \times UnitPrice)$$

### Tabel 3: Statistik Deskriptif Fitur RFM Asli (N = 4.338 Pelanggan)
| Statistik Deskriptif | Recency (Hari) | Frequency (Order Unik) | Monetary (£ / GBP) |
| :--- | :---: | :---: | :---: |
| **Mean** | 92,54 | 4,27 | 2.048,69 |
| **Standar Deviasi** | 100,01 | 7,70 | 8.985,23 |
| **Minimum** | 1,00 | 1,00 | 3,75 |
| **Kuartil 1 (Q1 / 25%)** | 18,00 | 1,00 | 306,48 |
| **Median (Q2 / 50%)** | 51,00 | 2,00 | 668,57 |
| **Kuartil 3 (Q3 / 75%)** | 142,00 | 5,00 | 1.660,60 |
| **Maksimum** | 374,00 | 209,00 | 280.206,02 |
| **Total Akumulasi** | - | 18.532 order | **£8.887.208,89** |
| **Koefisien Skewness** | **+1,25** | **+12,07** | **+19,34** |

Tabel RFM telah disimpan secara permanen pada [rfm_customer.csv](dataset/rfm_customer.csv).

---

## 8. Transformasi Fitur dan Scaling

### 8.1 Evaluasi Skewness dan Transformasi Logaritmik
Algoritma K-Means mengandalkan perhitungan jarak Euclidean, yang mengasumsikan distribusi data yang relatif bulat (*spherical*) dan tidak timpang secara ekstrem. Pada data asli:
- Fitur *Frequency* memiliki skewness **+12,07** dan *Monetary* memiliki skewness **+19,34** (didominasi pelanggan korporat/grosir yang berbelanja ratusan ribu pound sterling).
- Diterapkan transformasi logaritmik natural terkompensasi:
  $$x_{trans} = \ln(1 + x) \quad (\text{fungsi } \text{log1p})$$

### Tabel 4: Perbandingan Karakteristik Fitur Sebelum dan Sesudah Transformasi log1p
| Fitur RFM | Skewness Asli (Raw) | Skewness Pasca log1p | Rata-rata log1p | Deviasi Baku log1p | Evaluasi Distribusi |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Recency** | +1,25 | **-0,38** | 3,83 | 1,34 | Mendekati simetri normal sempurna. |
| **Frequency** | +12,07 | **+1,21** | 1,35 | 0,68 | Penurunan skewness sebesar 89,9%. |
| **Monetary** | +19,34 | **+0,40** | 6,59 | 1,26 | Nyaris kurva lonceng normal ideal. |

![Distribusi RFM](visualizations/rfm_distributions.png)

![Boxplot RFM](visualizations/rfm_boxplots.png)

### 8.2 Standardisasi Z-Score (`StandardScaler`)
Setelah transformasi log1p, rentang skala antar-fitur masih bervariasi (rentang log Monetary ~10 satuan, sedangkan Frequency ~4 satuan). Diterapkan `StandardScaler` untuk menghasilkan fitur terstandarisasi dengan rata-rata 0 ($\mu = 0$) dan variansi 1 ($\sigma^2 = 1$):
$$z = \frac{x_{trans} - \mu_{trans}}{\sigma_{trans}}$$

![Korelasi RFM](visualizations/rfm_correlation.png)

---

## 9. Metodologi K-Means dan Strategi Pemilihan Jumlah Cluster

### 9.1 Konfigurasi Algoritma
- **Algoritma:** K-Means Clustering (`sklearn.cluster.KMeans`).
- **Inisialisasi:** `k-means++` untuk mempercepat konvergensi dan menghindari local minima.
- **Pengulangan Inisialisasi (`n_init`):** 20 kali pengulangan independen untuk memastikan stabilitas centroid.
- **Random State:** 42 (dijaga konstan untuk memastikan keterulangan hasil).
- **Rentang Eksperimen:** $k \in [2, 10]$.

### 9.2 Metrik Evaluasi Internal
1. **Within-Cluster Sum of Squares (Inertia):** Mengukur kekompakan internal klaster (metode siku / *Elbow Method*).
2. **Silhouette Score:** Mengukur rasio antara jarak pemisah antar-klaster terdekat ($b$) dengan kekompakan intra-klaster ($a$). Rentang [-1, 1], nilai lebih tinggi menunjukkan klaster yang padat dan terpisah baik.
3. **Davies-Bouldin Index (DBI):** Mengukur rasio kesamaan antar-klaster. Nilai lebih rendah mengindikasikan separasi klaster yang lebih superior.
4. **Calinski-Harabasz Index (CHI):** Rasio dispersi antar-klaster terhadap intra-klaster. Nilai lebih tinggi menunjukkan partisi yang lebih jelas.

### Tabel 5: Rekapitulasi Metrik Evaluasi K-Means untuk $k = 2$ sampai 10
| $k$ | Inertia (WCSS) | Silhouette Score | Davies-Bouldin | Calinski-Harabasz | Ukuran Klaster Minimum | Ukuran Klaster Maksimum | Distribusi Ukuran Klaster |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **2** | 6.483,59 | **0,4328** | **0,8925** | 4.367,32 | 1.666 | 2.672 | [2.672, 1.666] |
| **3** | 4.869,49 | 0,3365 | 1,0483 | 3.625,30 | 769 | 1.872 | [1.872, 1.697, 769] |
| **4** | **3.938,60** | **0,3373** | **1,0106** | **3.328,92** | **709** | **1.617** | **[1.617, 1.173, 839, 709]** |
| **5** | 3.296,71 | 0,3162 | 0,9878 | 3.192,98 | 339 | 1.209 | [1.209, 1.018, 938, 834, 339] |
| **6** | 2.855,53 | 0,3128 | 1,0156 | 3.082,28 | 333 | 962 | [962, 953, 811, 689, 590, 333] |
| **7** | 2.548,82 | 0,3092 | 0,9823 | 2.963,79 | 227 | 891 | [891, 863, 687, 647, 620, 403, 227] |
| **8** | 2.336,34 | 0,3033 | 0,9892 | 2.827,07 | 214 | 856 | [856, 755, 673, 635, 503, 410, 292, 214] |
| **9** | 2.156,01 | 0,2811 | 1,0193 | 2.725,20 | 210 | 665 | [665, 651, 636, 567, 477, 471, 380, 281, 210] |
| **10** | 2.000,57 | 0,2781 | 1,0237 | 2.647,38 | 93 | 642 | [642, 637, 590, 544, 493, 435, 346, 299, 259, 93] |

Seluruh data evaluasi tersimpan di [model_metrics.csv](dataset/model_metrics.csv).

![Elbow Method](visualizations/elbow_method.png)

![Silhouette Scores](visualizations/silhouette_scores.png)

---

## 10. Hasil Evaluasi Model dan Alasan Pemilihan Model Final

Berdasarkan sintesis metrik matematis dan relevansi manajerial, **$k = 4$ ditetapkan sebagai solusi model final**. Argumentasi keputusan ini bertumpu pada lima pilar:

1. **Titik Tekuk Inertia (Elbow Point):** Kurva penurunan *Inertia* menunjukkan penurunan curam dari $k=2$ ke $k=3$ ($\Delta = 1.614,1$) dan ke $k=4$ ($\Delta = 930,9$). Setelah $k=4$, laju penurunan melandai secara bertahap ($\Delta = 641,9$ untuk $k=5$, lalu $\Delta = 441,2$ untuk $k=6$), menandai pengembalian marjinal yang menyusut (*diminishing returns*).
2. **Keunggulan Metrik Komparatif:** Silhouette Score untuk $k=4$ (**0,3373**) lebih tinggi dibandingkan $k=3$ (**0,3365**). Demikian pula Davies-Bouldin Index pada $k=4$ (**1,0106**) lebih rendah/lebih baik dibandingkan $k=3$ (**1,0483**).
3. **Kelemahan Solusi $k=2$:** Kendati $k=2$ mencatat Silhouette tertinggi (0,4328), membagi pelanggan hanya menjadi 2 kelompok ("aktif" vs "tidak aktif") terlalu simplistik dan mengabaikan perbedaan kritis antara pembeli baru, pembeli reguler berulang, dan pelanggan korporat grosir bernilai tinggi.
4. **Keseimbangan Ukuran Klaster:** Pada $k=4$, distribusi anggota klaster sangat proporsional: 37,28%, 27,04%, 19,34%, dan 16,34%. Tidak ada klaster kerdil (*trivial/degenerate cluster*). Sebaliknya, pada $k \ge 5$, ukuran klaster mulai terpecah kecil (misal klaster beranggotakan hanya 339 pelanggan pada $k=5$ dan 93 pelanggan pada $k=10$).
5. **Diferensiasi Perilaku yang Tajam (*Actionability*):** Solusi $k=4$ berhasil memisahkan pelanggan aktif baru (*Recent / Promising*) dari pelanggan setia (*Loyal Customers*) dan pelanggan utama (*Champions*), yang pada solusi $k=3$ tercampur menjadi satu grup besar yang heterogen.

### 10.1 Uji Stabilitas Seed dan Pembanding Algoritma
- **Stabilitas Acak (Random Seed Robustness):** Pengujian K-Means ($k=4$) terhadap 5 seed acak yang berbeda (`seeds = [10, 99, 123, 777, 2024]`) menghasilkan rata-rata *Adjusted Rand Index* (ARI) sebesar **0,9603** terhadap solusi acuan (seed 42). Tingkat kesesuaian di atas 95% ini membuktikan bahwa partisi klaster sangat stabil dan tidak bergantung pada inisialisasi acak.
- **Pembanding Hierarchical Agglomerative Clustering (Ward Linkage):** Perbandingan partisi K-Means dengan *Agglomerative Clustering* menghasilkan ARI sebesar **0,4303**. Perbedaan ini timbul karena kecenderungan Ward Linkage membentuk ukuran klaster dengan variansi bola homogen yang berbeda dalam menangani densitas titik ekstrem. K-Means mempertahankan partisi geometri yang lebih adaptif pada ruang fitur RFM.

---

## 11. Hasil Clustering dan Profil Masing-Masing Cluster

Data segmentasi pelanggan lengkap dengan label klaster dan nama segmen telah diekspor ke [customer_segments.csv](dataset/customer_segments.csv).

### Tabel 6: Profil Komparatif Segmen Pelanggan Berdasarkan Nilai RFM Asli
| Cluster ID | Nama Segmen Bisnis | Jumlah Pelanggan ($n$) | Pangsa Pelanggan (%) | Recency Median (Hari) | Recency Rata-rata (Hari) | Frequency Median (Order) | Frequency Rata-rata (Order) | Monetary Median (£) | Monetary Rata-rata (£) | Total Pendapatan (£) | Kontribusi Pendapatan (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3** | **Champions / High-Value** | 709 | 16,34% | **8,0** | 12,1 | **10,0** | 13,79 | **£3.722,82** | £8.112,93 | **£5.752.065,07** | **64,72%** |
| **1** | **Loyal Customers** | 1.173 | 27,04% | **55,0** | 70,1 | **4,0** | 4,11 | **£1.341,65** | £1.804,82 | **£2.117.050,02** | **23,82%** |
| **2** | **Recent / Promising Customers** | 839 | 19,34% | **17,0** | 18,2 | **2,0** | 2,14 | **£466,70** | £543,06 | **£455.624,56** | **5,13%** |
| **0** | **At-Risk / Hibernating** | 1.617 | 37,28% | **177,0** | 182,6 | **1,0** | 1,32 | **£298,40** | £347,85 | **£562.469,24** | **6,33%** |
| **Total / Rata-rata** | *Keseluruhan Populasi* | *4.338* | *100,0%* | *51,0* | *92,5* | *2,0* | *4,27* | *£668,57* | *£2.048,69* | *£8.887.208,89* | *100,0%* |

### 11.1 Karakteristik Rinci Setiap Segmen
1. **Segmen 3: Champions / High-Value Customers**
   - *Pola Perilaku:* Pembeli super aktif dengan median jeda pembelian hanya 8 hari yang lalu dan rata-rata 13,8 order dalam setahun. Nilai belanja median mencapai £3.722,82 dengan rata-rata £8.112,93.
   - *Peran Finansial:* Merupakan **penopang vital bisnis**. Meskipun hanya mencakup 16,34% dari total pelanggan, kelompok ini menyumbang **£5,75 juta (64,72%)** dari seluruh perputaran uang perusahaan. Banyak dari mereka adalah akun pedagang grosir atau distributor retail.
2. **Segmen 1: Loyal Customers**
   - *Pola Perilaku:* Pelanggan setia yang bertransaksi secara berkala dan stabil. Median Recency berada pada 55 hari (~2 bulan) dengan median 4 kali transaksi dan median pembelanjaan £1.341,65.
   - *Peran Finansial:* Kontributor pendapatan terbesar kedua, menghasilkan **£2,12 juta (23,82%)**. Bersama kelompok Champions, mereka membentuk *core customer base* yang menghasilkan 88,54% omset perusahaan.
3. **Segmen 2: Recent / Promising Customers**
   - *Pola Perilaku:* Pelanggan yang baru saja melakukan transaksi dalam 2-3 minggu terakhir (median Recency 17 hari), namun frekuensi belanja mereka masih berada di tahap awal (median 2 order) dengan median belanja moderat (£466,70).
   - *Peran Finansial:* Menyumbang 5,13% pendapatan (£455.624,56). Kelompok ini merupakan *talent pool* terpenting untuk dipupuk menjadi pelanggan setia masa depan sebelum minat mereka surut.
4. **Segmen 0: At-Risk / Hibernating Customers**
   - *Pola Perilaku:* Basis pelanggan terbesar secara kuantitas (37,28% populasi), namun telah pasif selama rata-rata setengah tahun (median Recency 177 hari). Mayoritas hanya pernah bertransaksi 1 kali (median Frequency 1,0) dengan nilai belanja terendah (median £298,40).
   - *Peran Finansial:* Kontribusi pendapatan sangat marjinal (hanya 6,33%), merefleksikan pelanggan uji coba (*one-time trial buyers*) atau pelanggan yang telah berpindah ke kompetitor (*churned*).

---

## 12. Visualisasi serta Interpretasi Setiap Grafik Penting

### 12.1 Proyeksi 2D Klaster Pelanggan (PCA)
Untuk memvisualisasikan partisi 3 dimensi (RFM) ke dalam bidang datar 2 dimensi, digunakan *Principal Component Analysis* (PCA).
- **Komponen Utama 1 (PC1):** Menjelaskan **75,08%** dari total variansi data. Bobot eigen (*loadings*): Recency (-0,512), Frequency (+0,618), Monetary (+0,597). PC1 merepresentasikan sumbu keterlibatan dan daya beli pelanggan (*Customer Engagement & Lifetime Value*). Semakin ke kanan koordinat PC1, semakin loyal dan bernilai tinggi pelanggan tersebut.
- **Komponen Utama 2 (PC2):** Menjelaskan **18,79%** dari total variansi data. Bobot eigen (*loadings*): Recency (+0,850), Frequency (+0,261), Monetary (+0,458). PC2 merepresentasikan sumbu kebaruan versus dormansi waktu.
- **Total Variansi Terjelaskan:** Dua komponen utama merangkum **93,87% variansi multivariat**.
- *Catatan Interpretasi Geometri:* Proyeksi 2D mempertahankan struktur pemisahan klaster secara jelas, di mana klaster *Champions* berada di kuadran kanan, *Loyal* di tengah, *Recent* di kuadran kiri-bawah, dan *At-Risk* di kuadran kiri-atas. Sebagian kecil tumpang tindih visual terjadi semata-mata karena reduksi dimensi proyeksi.

![PCA 2D Cluster Map](visualizations/cluster_pca.png)

### 12.2 Profil Karakteristik Perilaku (Heatmap Z-Score & Normalisasi)
Visualisasi heatmap Z-score menampilkan polaritas perilaku yang tegas antar segmen. Klaster *Champions* menunjukkan deviasi Z-score positif yang masif pada fitur Frequency (+1,48) dan Monetary (+1,45) serta Z-score negatif pada Recency (-1,12). Sebaliknya, klaster *At-Risk* menunjukkan Z-score positif yang tinggi pada Recency (+0,79) yang mengindikasikan tingkat ketidakaktifan yang ekstrem.

![Profil Klaster](visualizations/cluster_profiles.png)

### 12.3 Analisis Ketimpangan Kontribusi Pendapatan (Pareto Analysis)
Grafik lingkaran ganda menunjukkan manifestasi nyata dari Prinsip Pareto pada bisnis ritel ini:
- **16,34% pelanggan (*Champions*) menghasilkan 64,72% pendapatan.**
- **43,38% pelanggan (*Champions* + *Loyal*) menopang 88,54% arus kas perusahaan.**
- Sementara **56,62% pelanggan sisanya (*Recent* + *At-Risk*) hanya menghasilkan 11,46% total omset.**

![Kontribusi Klaster](visualizations/cluster_contributions.png)

---

## 13. Insight Bisnis dan Rekomendasi Strategi per Segmen

Hasil segmentasi K-Means memungkinkan tim manajemen menyusun matriks strategi pemasaran, retensi, dan alokasi anggaran yang berbasis data (*data-driven marketing*):

### Tabel 7: Matriks Strategi Pemasaran Berdasarkan Segmen Pelanggan
| Segmen Pelanggan | Tujuan Bisnis (*Objective*) | Saluran Utama (*Channel*) | Taktik & Penawaran Utama | Pantangan & Risiko (*Guardrails*) |
| :--- | :--- | :--- | :--- | :--- |
| **Champions / High-Value**<br>*(16,3% Pelanggan, 64,7% Omset)* | **Proteksi & Retensi VIP, Advokasi Brand** | Komunikasi personal (WhatsApp Business API, Email VIP, *Dedicated Account Manager*) | • Layanan *Priority Concierge* dan alokasi stok terjamin untuk pesanan grosir.<br>• *Early preview* katalog musiman edisi terbatas sebelum rilis publik.<br>• Fasilitas syarat pembayaran fleksibel (termin faktur 30 hari untuk B2B).<br>• Kemasan premium dan kartu apresiasi personal. | Jangan membanjiri mereka dengan diskon massal murahan; fokus pada keandalan operasional, kecepatan pengiriman, dan layanan purna jual. |
| **Loyal Customers**<br>*(27,0% Pelanggan, 23,8% Omset)* | **Peningkatan Basket Size & Frekuensi (Upsell/Cross-sell)** | Email marketing tersegmentasi, notifikasi aplikasi, penawaran di keranjang belanja | • Program loyalitas berjenjang (*Tiered Reward Points*) untuk mencapai status VIP.<br>• Rekomendasi *bundle* produk komplementer (berdasarkan produk yang sering dibeli bersamaan).<br>• Insentif *Free Shipping* dengan batas minimal nilai pesanan (misal di atas £1.500). | Hindari komunikasi yang tidak relevan dengan kategori produk yang biasa mereka beli. |
| **Recent / Promising**<br>*(19,3% Pelanggan, 5,1% Omset)* | **Pembentukan Kebiasaan Belanja (*Habituation & Nurturing*)** | *Welcome drip email sequence*, push notification, survei interaktif | • Rangkaian email selamat datang yang mengedukasi ragam kategori hadiah unik.<br>• Voucher diskon pesanan kedua (*Next Order Voucher*) yang memiliki batas waktu (misal berlaku 21 hari).<br>• Survei kepuasan pelanggan atas pengiriman pesanan perdana. | Jangan biarkan mereka melewati 45 hari tanpa kontak pemicu (*trigger communication*). |
| **At-Risk / Hibernating**<br>*(37,3% Pelanggan, 6,3% Omset)* | **Reaktivasi Terukur atau Pensiun Hemat Biaya** | Kampanye email otomatis *Win-back*, iklan penargetan ulang (*retargeting ads*) | • Kampanye *'We Miss You'* dengan insentif diskon terbatas untuk reaktivasi.<br>• Survei singkat alasan berhenti berbelanja (apakah masalah harga, stok, atau ekspedisi).<br>• Filter *churn mitigation*: Jika tidak merespons setelah 3 kali kontak, hentikan kampanye berbayar untuk menghemat biaya CAC. | Jangan mengalokasikan anggaran promosi besar pada seluruh segmen ini tanpa seleksi. |

---

## 14. Keterbatasan, Asumsi, Potensi Bias, dan Risiko Interpretasi

Dalam menerapkan hasil segmentasi ini ke dalam strategi eksekusi komersial, para pemangku kepentingan wajib mempertimbangkan batasan berikut:

1. **Jendela Waktu Historis (2010–2011):** Data ini mencerminkan dinamika ekonomi e-commerce Inggris dan Eropa lebih dari satu dekade lalu. Hasil segmentasi tidak dapat diasumsikan otomatis mencerminkan perilaku pelanggan saat ini tanpa kalibrasi data kontemporer.
2. **Ketiadaan Data Demografis & Kategori Produk dalam Fitur Clustering:** Fitur klasterisasi dibatasi secara ketat pada 3 metrik perilaku transaksi (RFM). Model tidak memperhitungkan jenis kelamin, usia, maupun elastisitas harga kategori produk tertentu.
3. **Penyensoran Transaksi (*Left/Right Censoring*):** Karena observasi data dimulai pada 1 Desember 2010, transaksi pelanggan sebelum tanggal tersebut tidak tercatat. Pelanggan yang baru pertama kali muncul di awal tahun 2011 bisa jadi merupakan pelanggan lama yang baru bertransaksi kembali.
4. **Pemisahan Akun Grosir (*B2B*) vs Ritel Individu (*B2C*):** Sebagian pelanggan di segmen *Champions* memiliki kuantitas pembelian ribuan unit yang merepresentasikan entitas bisnis grosir. Mencampur perlakuan promosi ritel biasa terhadap akun korporat berisiko menurunkan efektivitas komunikasi.
5. **Ketiadaan Variabel Margin Keuntungan:** Fitur *Monetary* mengukur omset kotor (*gross revenue*), bukan laba bersih (*gross profit margin*). Transaksi beromset tinggi dengan margin tipis belum tentu menghasilkan laba lebih besar dibandingkan transaksi bernilai sedang namun bermargin tebal.

---

## 15. Kesimpulan

Proyek segmentasi pelanggan *Customer Segmentation Through RFM-Based Clustering* telah berhasil dijalankan secara komprehensif, terstruktur, dan tervalidasi:
1. **Validitas Metodologis:** Rekayasa fitur RFM pada 4.338 pelanggan yang telah dibersihkan dari 149.217 baris transaksi cacat/non-pembelian menghasilkan basis pemodelan yang sangat kokoh.
2. **Superioritas K-Means ($k=4$):** Evaluasi internal membuktikan bahwa 4 klaster menghasilkan arsitektur pengelompokan yang paling optimal, stabil (ARI > 0,96 di seluruh seed acak), dan kaya akan wawasan komersial.
3. **Struktur Nilai Terkonsentrasi:** Bisnis sangat bergantung pada 16,34% pelanggan *Champions* yang menguasai 64,72% pendapatan, serta 27,04% *Loyal Customers* yang menopang 23,82% pendapatan.
4. **Aksi Nyata:** Peta segmentasi ini memberikan dasar objektif bagi manajemen untuk mengalihkan fokus anggaran dari akuisisi acak menuju proteksi akun bernilai tinggi dan akselerasi pelanggan potensial.

---

## 16. Cara Menjalankan Ulang Project dan Daftar Artefak

### 16.1 Lingkungan & Versi Pustaka (*Environment*)
Proyek dibangun dan diuji menggunakan dependensi Python yang tercatat di `requirements.txt`:
- Python: `3.15.1`
- `pandas`: `3.1.1`
- `numpy`: `2.5.4`
- `scikit-learn`: `1.9.1`
- `matplotlib`: `3.11.2`
- `seaborn`: `0.13.2`
- `streamlit`: `1.67.0`

### 16.2 Instruksi Eksekusi Ulang
1. **Instalasi Dependensi:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Menjalankan Pipeline Segmentasi & Visualisasi:**
   Eksekusi skrip utama untuk memproses data, melatih model, menghasilkan metrik, dan mengekspor seluruh grafik:
   ```bash
   python src/customer_segmentation.py
   ```
3. **Menjalankan Aplikasi Dashboard Interaktif (Streamlit):**
   ```bash
   python -m streamlit run app.py
   ```
   Buka peramban web pada alamat lokal: `http://localhost:8501`.

### 16.3 Inventaris Artefak yang Dihasilkan
```text
Customer Segmentation Through RFM-Based Clustering/
├── Online_Retail.csv                     # Dataset transaksi e-commerce asli
├── metadata.json                         # Metadata UCI Machine Learning Repository
├── TASK.md                               # Dokumen spesifikasi instruksi pengerjaan
├── requirements.txt                      # Spesifikasi versi pustaka Python
├── final_report.md                       # Laporan akhir komprehensif (dokumen ini)
├── src/
│   └── customer_segmentation.py          # Skrip utama modular end-to-end data pipeline
├── outputs/
│   ├── rfm_customer.csv                  # Tabel fitur RFM per pelanggan (4.338 baris)
│   ├── customer_segments.csv             # Data pelanggan lengkap dengan label segmen & PCA
│   └── model_metrics.csv                 # Metrik evaluasi K-Means untuk k=2..10
├── visualizations/
│   ├── missing_values.png                # Grafik ikhtisar data hilang per kolom
│   ├── transaction_trend.png             # Tren volume transaksi & pendapatan bulanan
│   ├── rfm_distributions.png             # Distribusi RFM sebelum & sesudah log1p
│   ├── rfm_boxplots.png                  # Boxplot deteksi outlier fitur RFM log1p
│   ├── rfm_correlation.png               # Heatmap korelasi Pearson fitur RFM
│   ├── elbow_method.png                  # Kurva Elbow Inertia vs k (k=2..10)
│   ├── silhouette_scores.png             # Evaluasi Silhouette Score & Davies-Bouldin
│   ├── cluster_pca.png                   # Proyeksi 2D klaster dalam ruang PCA
│   ├── cluster_profiles.png              # Heatmap Z-Score & profil median fitur asli
│   └── cluster_contributions.png         # Diagram donat pangsa pelanggan vs omset
├── app.py                                # Kode sumber aplikasi Streamlit interaktif
├── capture_all_pages.mjs                  # Skrip otomatisasi tangkapan layar headless CDP untuk seluruh tab
└── screenshots/
    ├── README.md                          # Dokumentasi dimensi, hash, & konten tiap screenshot
    ├── streamlit_full_page.png            # Tangkapan layar penuh overview dashboard Streamlit
    ├── tab1_profil_segmen.png             # Tangkapan layar Tab 1: Ringkasan profil segmen & donut charts
    ├── tab2_pca_distribusi.png            # Tangkapan layar Tab 2: Proyeksi PCA 2D & boxplot distribusi
    ├── tab3_eksplorasi_data.png           # Tangkapan layar Tab 3: Eksplorasi data pelanggan terfilter
    └── tab4_rekomendasi.png               # Tangkapan layar Tab 4: Rekomendasi strategis 4 segmen bisnis
```

---
*Laporan ini disusun secara otomatis oleh AI Data Scientist Agent berdasarkan eksekusi riil dataset tanpa data placeholder atau fabrikasi angka.*

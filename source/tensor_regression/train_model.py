import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

data = pd.read_csv('../../library/optimized/full_dataset.csv')

print(data['salary'])

data.fillna('', inplace=True)

perks = data['perks'].str.split(',').explode().unique()

data['perks'] = data['perks'].str.split(',').apply(lambda x: [s.strip() for s in x])

unique_vals = set(val for sublist in data['perks'] for val in sublist)

for val in unique_vals:
    data[val] = 0

# for index, row in data.iterrows():
#     for val in row['perks']:
#         data.at[index, val] = 1

data.drop(columns=['perks'], inplace=True)

print(data['salary'])
print(data)



data = data.drop('id', axis=1)
data = data.copy()
data = pd.get_dummies(data)
print(data)

sorted_cols = sorted(data.columns)
X = data[sorted_cols]
X = X.drop('salary', axis=1)
# print(X.columns.tolist())
print(X.columns.tolist())
# print(X)
X = np.array(X)
# print(X) 166

y = data['salary']
print(y)
y = np.array(y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# print(X_train)

print("importing tensorflow...")
import tensorflow as tf
print("tensorflow loaded. version: ", tf.__version__)

X = tf.convert_to_tensor(X, dtype=tf.float64)
y = tf.convert_to_tensor(y, dtype=tf.float64)

dropout_val = 0.2
density = 128

l1 = 0.0
l2 = 0.0
reg = tf.keras.regularizers.l1_l2(l1=l1, l2=l2)
act_type = 'relu'

savename = 'b32_l4_d128_r0-2_l1l2-0_0.png'


model = tf.keras.Sequential([
    tf.keras.layers.Dense(density, activation=act_type, kernel_regularizer=reg, input_shape=(X.shape[1],)),
    tf.keras.layers.Dropout(dropout_val),
    tf.keras.layers.Dense(density, activation=act_type, kernel_regularizer=reg),
    tf.keras.layers.Dropout(dropout_val),
    tf.keras.layers.Dense(density, activation=act_type, kernel_regularizer=reg),
    tf.keras.layers.Dropout(dropout_val),
    tf.keras.layers.Dense(density, activation=act_type, kernel_regularizer=reg),
    tf.keras.layers.Dropout(dropout_val),
    tf.keras.layers.Dense(1)
])

loss_fn = tf.keras.losses.MeanSquaredError()

model.compile(
    loss=loss_fn,
    optimizer='adam'
)

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(density, activation=act_type, kernel_regularizer=reg, input_shape=(X.shape[1],)),
#     tf.keras.layers.Dropout(dropout_val),
#     tf.keras.layers.Dense(density, activation=act_type, kernel_regularizer=reg),
#     tf.keras.layers.Dropout(dropout_val),
#     tf.keras.layers.Dense(density, activation=act_type, kernel_regularizer=reg),
#     tf.keras.layers.Dropout(dropout_val),
#     tf.keras.layers.Dense(density, activation=act_type, kernel_regularizer=reg),
#     tf.keras.layers.Dropout(dropout_val),
#     tf.keras.layers.Dense(density, activation=act_type, kernel_regularizer=reg),
#     tf.keras.layers.Dropout(dropout_val),
#     tf.keras.layers.Dense(density, activation=act_type, kernel_regularizer=reg),
#     tf.keras.layers.Dropout(dropout_val),
#     tf.keras.layers.Dense(density, activation=act_type, kernel_regularizer=reg),
#     tf.keras.layers.Dropout(dropout_val),
#     tf.keras.layers.Dense(density, activation=act_type, kernel_regularizer=reg),
#     tf.keras.layers.Dropout(dropout_val),
#     tf.keras.layers.Dense(density, activation=act_type, kernel_regularizer=reg),
#     tf.keras.layers.Dropout(dropout_val),
#     tf.keras.layers.Dense(density, activation=act_type, kernel_regularizer=reg),
#     tf.keras.layers.Dropout(dropout_val),
#     tf.keras.layers.Dense(density, activation=act_type, kernel_regularizer=reg),
#     tf.keras.layers.Dropout(dropout_val),
#     tf.keras.layers.Dense(density, activation=act_type, kernel_regularizer=reg),
#     tf.keras.layers.Dropout(dropout_val),
#     tf.keras.layers.Dense(density, activation=act_type, kernel_regularizer=reg),
#     tf.keras.layers.Dropout(dropout_val),
#     tf.keras.layers.Dense(density, activation=act_type, kernel_regularizer=reg),
#     tf.keras.layers.Dropout(dropout_val),
#     tf.keras.layers.Dense(density, activation=act_type, kernel_regularizer=reg),
#     tf.keras.layers.Dropout(dropout_val),
#     tf.keras.layers.Dense(density, activation=act_type, kernel_regularizer=reg),
#     tf.keras.layers.Dense(1)
# ])

import time
start_time = time.time()

class TimeLimitCallback(tf.keras.callbacks.Callback):
    def __init__(self, time_limit):
        super(TimeLimitCallback, self).__init__()
        self.time_limit = time_limit
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if time.time() - self.start_time >= self.time_limit:
            self.model.stop_training = True
            print(f'\nTime limit of {time.time() - self.start_time} seconds reached. Stopping training...')

import matplotlib.pyplot as plt
history = model.fit(
    X, 
    y, 
    epochs=200, 
    batch_size=32, 
    validation_split=0.2,
    callbacks=[
        TimeLimitCallback(60 * 60 * 1),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=25,
            verbose=1,
            restore_best_weights=True
        )
    ]
)

train_loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(train_loss))
plt.plot(epochs, train_loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(savename)
plt.show()

end_time = time.time()
print("Training time: ", end_time - start_time)

model.evaluate(X, y, verbose=2)
model.save('last_test.keras')

Complete_Data = pd.read_csv('../../library/optimized/full_dataset.csv')
Complete_Data.fillna('', inplace=True)
Complete_Data = pd.get_dummies(Complete_Data)
complete_data_cols = ['', '-', 'Annual Reward', 'Asuransi Gigi', 'Asuransi kesehatan', 'BPJS', 'BPJS Kesehatan', 'BPJS Kesehatan dan BPJS Ketenagakerjaan', 'BPJS Ketenagakerjaan', 'Bisnis (contoh: Kemeja)', 'Bonus', 'Business', 'Company Uniform', 'Formal Casual', 'Formil (contoh: Kemeja + Dasi)', 'Jam Bekerja yang Panjang', 'Jamsostek', 'Kasual (contoh: Kaos)', 'Ketenagakerjaan', 'Lunch BPJS Career Path', 'Lunch and Private Insurance', 'Monday - Friday', 'Monday - Saturday', 'Monday to Saturday', 'Monday-Friday', 'Monday-Saturday', 'Mondays - Saturdays', 'Mondays-Saturdays', 'Olahraga (contoh: pusat kebugaran)', 'Other Benefits', 'Others Benefits', 'Parkir', 'Penglihatan', 'Pinjaman', 'Scheduled performance appraisals', 'Senin - Jumat', 'Senin - Sabtu', 'Senin-Sabtu', 'Seragam', 'Smart Casual', 'Smart Casual Business', 'Tidy casual', 'Tip', 'Transportasi', 'Tunjangan Pendidikan', 'Uniform', 'Waktu regular', 'e_d3', 'e_d4', 'e_dp', 'e_gp', 'e_none', 'e_s1', 'e_s2', 'e_s3', 'e_sma', 'e_smu/smk/stm', 'e_sp', 'experience_', 'experience_1.0', 'experience_10.0', 'experience_11.0', 'experience_12.0', 'experience_15.0', 'experience_17.0', 'experience_2.0', 'experience_20.0', 'experience_3.0', 'experience_4.0', 'experience_5.0', 'experience_6.0', 'experience_7.0', 'experience_8.0', 'gig1_Aktuaria/Statistik', 'gig1_Akuntansi Umum / Pembiayaan', 'gig1_Angkatan Bersenjata', 'gig1_Arsitek/Desain Interior', 'gig1_Audit & Pajak', 'gig1_Biomedis', 'gig1_Bioteknologi', 'gig1_Diagnosa/Lainnya', 'gig1_Digital Marketing', 'gig1_E-commerce', 'gig1_Farmasi', 'gig1_Geologi/Geofisika', 'gig1_Hiburan', 'gig1_Hotel/Pariwisata', 'gig1_Hubungan Masyarakat', 'gig1_IT-Admin Jaringan/Sistem/Database', 'gig1_IT-Perangkat Keras', 'gig1_IT-Perangkat Lunak', 'gig1_Jurnalis/Editor', 'gig1_Keuangan / Investasi Perusahaan ', 'gig1_Kimia', 'gig1_Kontrol Proses', 'gig1_Lainnya/Kategori tidak tersedia', 'gig1_Layanan Pelanggan', 'gig1_Layanan Sosial/Konseling', 'gig1_Logistik/Rantai Pasokan', 'gig1_Makanan/Minuman/Pelayanan Restoran', 'gig1_Manufaktur', 'gig1_Mekanikal', 'gig1_Merchandising', 'gig1_Minyak/Gas', 'gig1_Pekerjaan Umum', 'gig1_Pelatihan & Pengembangan', 'gig1_Pemasaran/Pengembangan Bisnis', 'gig1_Pembelian/Manajemen Material', 'gig1_Pemeliharaan', 'gig1_Pendidikan', 'gig1_Penerbitan', 'gig1_Pengacara / Asisten Legal', 'gig1_Penjaminan Kualitas / QA', 'gig1_Penjualan - Jasa Keuangan', 'gig1_Penjualan - Korporasi', 'gig1_Penjualan - Teknik/Teknikal/IT', 'gig1_Penjualan Ritel', 'gig1_Perawatan Pribadi', 'gig1_Perbankan / Jasa Finansial ', 'gig1_Periklanan', 'gig1_Pertanian', 'gig1_Praktisi/Asisten Medis', 'gig1_Properti/Real Estate', 'gig1_Sains & Teknologi', 'gig1_Sekretaris', 'gig1_Seni / Desain Kreatif', 'gig1_Staf / Administrasi umum', 'gig1_Sumber Daya Manusia / HR', 'gig1_Survei Kuantitas', 'gig1_Teknik Elektro', 'gig1_Teknik Elektronika', 'gig1_Teknik Industri', 'gig1_Teknik Kimia', 'gig1_Teknik Lainnya', 'gig1_Teknik Lingkungan', 'gig1_Teknik Sipil/Konstruksi Bangunan', 'gig1_Teknikal & Bantuan Pelanggan', 'gig1_Teknologi Makanan/Ahli Gizi', 'gig1_Telesales/Telemarketing', 'gig1_Top Management / Manajemen Tingkat Atas', 'gig2_Akuntansi / Keuangan', 'gig2_Bangunan/Konstruksi', 'gig2_Hotel/Restoran', 'gig2_Komputer/Teknologi Informasi', 'gig2_Lainnya', 'gig2_Layanan Kesehatan', 'gig2_Manufaktur', 'gig2_Pelayanan', 'gig2_Pendidikan/Pelatihan', 'gig2_Penjualan / Pemasaran', 'gig2_Sains', 'gig2_Seni/Media/Komunikasi', 'gig2_Sumber Daya Manusia/Personalia', 'gig2_Teknik', 'industry_', 'industry_Agrikultural/Perkebunan/Peternakan Unggas/Perikanan', 'industry_Akunting / Audit / Layanan Pajak', 'industry_Asuransi', 'industry_Automobil/Mesin Tambahan Automotif/Kendaraan', 'industry_Bahan Kimia/Pupuk/Pestisida', 'industry_BioTeknologi/Farmasi/Riset klinik', 'industry_Call Center/IT-Enabled Services/BPO', 'industry_Elektrikal & Elektronik', 'industry_Hiburan/Media', 'industry_Hotel/Pariwisata', 'industry_Hukum/Legal', 'industry_Ilmu Pengetahuan & Teknologi', 'industry_Industri Berat/Mesin/Peralatan', 'industry_Jual Beli Saham/Sekuritas', 'industry_Jurnalisme', 'industry_Kayu/Fiber/Kertas', 'industry_Keamanan/Penegak hukum', 'industry_Kelautan/Aquakultur', 'industry_Kesehatan/Medis', 'industry_Komputer/Teknik Informatika (Perangkat Keras)', 'industry_Komputer/Teknik Informatika (Perangkat Lunak)', 'industry_Konstruksi/Bangunan/Teknik', 'industry_Konsultasi (Bisnis & Manajemen)', 'industry_Konsultasi (IT, Ilmu Pengetahuan, Teknis & Teknikal)', 'industry_Lainnya', 'industry_Layanan Umum/Tenaga Penggerak', 'industry_Lingkungan/Kesehatan/Keamanan', 'industry_Makanan & Minuman/Katering/Restoran', 'industry_Manajemen/Konsulting HR', 'industry_Manufaktur/Produksi', 'industry_Minyak/Gas/Petroleum', 'industry_Olahraga', 'industry_Organisasi Nirlaba/Pelayanan Sosial/LSM', 'industry_Pakaian', 'industry_Pameran/Manajemen acara/PIKP', 'industry_Pelayanan Arsitek/Desain Interior', 'industry_Pelayanan Perbaikan & Pemeliharaan', 'industry_Pemerintahan/Pertahanan', 'industry_Pendidikan', 'industry_Perawatan/Kecantikan/Fitnes', 'industry_Perbankan/Pelayanan Keuangan', 'industry_Percetakan/Penerbitan', 'industry_Periklanan/Marketing/Promosi/Hubungan Masyarakat', 'industry_Permata/Perhiasan', 'industry_Pertambangan', 'industry_Polymer/Plastik/Karet/Ban', 'industry_Produk Konsumen/Barang konsumen yang bergerak cepat', 'industry_Properti/Real Estate', 'industry_R&D', 'industry_Retail/Merchandise', 'industry_Seni/Desain/Fashion', 'industry_Tekstil/Garment', 'industry_Telekomunikasi', 'industry_Transportasi/Logistik', 'industry_Travel/Pariwisata', 'industry_Umum & Grosir', 'location_Aceh', 'location_Ambon', 'location_Badung', 'location_Bali', 'location_Balikpapan', 'location_Bandar Lampung', 'location_Bandung', 'location_Bangka', 'location_Bangka Belitung', 'location_Banjar', 'location_Banjarbaru', 'location_Banjarmasin', 'location_Banjarnegara', 'location_Banten', 'location_Bantul', 'location_Banyuwangi', 'location_Batam', 'location_Bekasi', 'location_Belitung', 'location_Bengkulu', 'location_Binjai', 'location_Bintan', 'location_Blitar', 'location_Bogor', 'location_Brebes', 'location_Bukittinggi', 'location_Cianjur', 'location_Cibinong', 'location_Cikarang', 'location_Cikupa', 'location_Cilacap', 'location_Cilegon', 'location_Cileungsi', 'location_Cimahi', 'location_Cirebon', 'location_Citeureup', 'location_Denpasar', 'location_Depok', 'location_Dumai', 'location_Gianyar', 'location_Gorontalo', 'location_Gowa', 'location_Gresik', 'location_Halmahera', 'location_Hulu Sungai Tengah', 'location_Jakarta Barat', 'location_Jakarta Pusat', 'location_Jakarta Raya', 'location_Jakarta Selatan', 'location_Jakarta Timur', 'location_Jakarta Utara', 'location_Jambi', 'location_Jawa Barat', 'location_Jawa Tengah', 'location_Jawa Timur', 'location_Jayapura', 'location_Jember', 'location_Jepara', 'location_Kalimantan Barat', 'location_Kalimantan Selatan', 'location_Kalimantan Tengah', 'location_Kalimantan Timur', 'location_Kalimantan Utara', 'location_Kapuas', 'location_Karawang', 'location_Kediri', 'location_Kendari', 'location_Kepulauan Riau', 'location_Kepulauan Seribu', 'location_Ketapang', 'location_Klaten', 'location_Klungkung', 'location_Kota Banda Aceh', 'location_Kotabaru', 'location_Kotawaringin Timur', 'location_Kudus', 'location_Kupang', 'location_Kuta', 'location_Kutai Timur', 'location_Lampung', 'location_Madiun', 'location_Madura', 'location_Magelang', 'location_Makassar', 'location_Malang', 'location_Maluku', 'location_Maluku Utara', 'location_Mamuju', 'location_Manado', 'location_Maros', 'location_Mataram', 'location_Medan', 'location_Metro', 'location_Minahasa', 'location_Mojokerto', 'location_Muara Enim', 'location_Nunukan', 'location_Nusa Tenggara Timur', 'location_Padang', 'location_Palangkaraya', 'location_Palembang', 'location_Palopo', 'location_Palu', 'location_Pandeglang', 'location_Pangkal Pinang', 'location_Papua', 'location_Pare-Pare', 'location_Pasuruan', 'location_Pekalongan', 'location_Pekanbaru', 'location_Pemalang', 'location_Penajam Paser Utara', 'location_Ponorogo', 'location_Pontianak', 'location_Poso', 'location_Prabumulih', 'location_Purwakarta', 'location_Purwokerto', 'location_Purworejo', 'location_Rangkasbitung', 'location_Riau', 'location_Salatiga', 'location_Samarinda', 'location_Semarang', 'location_Seminyak', 'location_Serang', 'location_Sibolga', 'location_Sidoarjo', 'location_Singkawang', 'location_Sleman', 'location_Sukabumi', 'location_Sulawesi Barat', 'location_Sulawesi Selatan', 'location_Sulawesi Tengah', 'location_Sulawesi Tenggara', 'location_Sulawesi Utara', 'location_Sumatera Barat', 'location_Sumatera Selatan', 'location_Sumatera Utara', 'location_Surabaya', 'location_Surakarta', 'location_Tanah Bumbu', 'location_Tangerang', 'location_Tanjung Balai', 'location_Tarakan', 'location_Tasikmalaya', 'location_Tegal', 'location_Ternate', 'location_Timika', 'location_Tuban', 'location_Ubud', 'location_Ungaran', 'location_Wonogiri ', 'location_Yogyakarta', 'position_CEO/GM/Direktur/Manajer Senior', 'position_Lulusan baru/Pengalaman kerja kurang dari 1 tahun', 'position_Manajer/Asisten Manajer', 'position_Pegawai (non-manajemen & non-supervisor)', 'position_Supervisor/Koordinator', 'senin - sabtu', 'size_', 'size_1000.0', 'size_200.0', 'size_2000.0', 'size_50.0', 'size_500.0', 'size_5000.0', 'turn_time_', 'turn_time_1.0', 'turn_time_10.0', 'turn_time_11.0', 'turn_time_12.0', 'turn_time_13.0', 'turn_time_14.0', 'turn_time_15.0', 'turn_time_16.0', 'turn_time_17.0', 'turn_time_18.0', 'turn_time_19.0', 'turn_time_2.0', 'turn_time_20.0', 'turn_time_21.0', 'turn_time_22.0', 'turn_time_23.0', 'turn_time_24.0', 'turn_time_25.0', 'turn_time_26.0', 'turn_time_27.0', 'turn_time_28.0', 'turn_time_29.0', 'turn_time_3.0', 'turn_time_30.0', 'turn_time_4.0', 'turn_time_5.0', 'turn_time_6.0', 'turn_time_7.0', 'turn_time_8.0', 'turn_time_9.0', 'type_Kontrak', 'type_Magang', 'type_Paruh Waktu', 'type_Penuh Waktu', 'type_Penuh Waktu, Kontrak', 'type_Penuh Waktu, Magang', 'type_Penuh Waktu, Paruh Waktu', 'type_Temporer', 'uniform']
Complete_y = Complete_Data['salary']
data_cols = data.columns.tolist()

missing_cols = set(complete_data_cols) - set(Complete_Data.columns.tolist())
for col in missing_cols:
    Complete_Data[col] = 0

excess_cols = set(Complete_Data.columns.tolist()) - set(complete_data_cols)
for col in excess_cols:
    Complete_Data.drop(col, axis=1, inplace=True)

sorted_cols = sorted(Complete_Data.columns)

Complete_X = Complete_Data[sorted_cols]
print(len(Complete_X.columns.tolist()))
Complete_X = np.array(Complete_X)
Complete_X = tf.convert_to_tensor(Complete_X, dtype=tf.float64)

prediction = model.predict(Complete_X)
aLLPRED = model.predict(X)
# print(prediction)

from sklearn.metrics import mean_squared_error
print(mean_squared_error(Complete_y, prediction))
print(mean_squared_error(y, aLLPRED))
print(mean_squared_error(prediction, prediction))





import csv

final_data = pd.read_csv('../../library/optimized/predict_dataset.csv')
final_data.fillna('', inplace=True)
final_data = pd.get_dummies(final_data)

train_cols = ['', '-', 'Annual Reward', 'Asuransi Gigi', 'Asuransi kesehatan', 'BPJS', 'BPJS Kesehatan', 'BPJS Kesehatan dan BPJS Ketenagakerjaan', 'BPJS Ketenagakerjaan', 'Bisnis (contoh: Kemeja)', 'Bonus', 'Business', 'Company Uniform', 'Formal Casual', 'Formil (contoh: Kemeja + Dasi)', 'Jam Bekerja yang Panjang', 'Jamsostek', 'Kasual (contoh: Kaos)', 'Ketenagakerjaan', 'Lunch BPJS Career Path', 'Lunch and Private Insurance', 'Monday - Friday', 'Monday - Saturday', 'Monday to Saturday', 'Monday-Friday', 'Monday-Saturday', 'Mondays - Saturdays', 'Mondays-Saturdays', 'Olahraga (contoh: pusat kebugaran)', 'Other Benefits', 'Others Benefits', 'Parkir', 'Penglihatan', 'Pinjaman', 'Scheduled performance appraisals', 'Senin - Jumat', 'Senin - Sabtu', 'Senin-Sabtu', 'Seragam', 'Smart Casual', 'Smart Casual Business', 'Tidy casual', 'Tip', 'Transportasi', 'Tunjangan Pendidikan', 'Uniform', 'Waktu regular', 'e_d3', 'e_d4', 'e_dp', 'e_gp', 'e_none', 'e_s1', 'e_s2', 'e_s3', 'e_sma', 'e_smu/smk/stm', 'e_sp', 'experience_', 'experience_1.0', 'experience_10.0', 'experience_11.0', 'experience_12.0', 'experience_15.0', 'experience_17.0', 'experience_2.0', 'experience_20.0', 'experience_3.0', 'experience_4.0', 'experience_5.0', 'experience_6.0', 'experience_7.0', 'experience_8.0', 'gig1_Aktuaria/Statistik', 'gig1_Akuntansi Umum / Pembiayaan', 'gig1_Angkatan Bersenjata', 'gig1_Arsitek/Desain Interior', 'gig1_Audit & Pajak', 'gig1_Biomedis', 'gig1_Bioteknologi', 'gig1_Diagnosa/Lainnya', 'gig1_Digital Marketing', 'gig1_E-commerce', 'gig1_Farmasi', 'gig1_Geologi/Geofisika', 'gig1_Hiburan', 'gig1_Hotel/Pariwisata', 'gig1_Hubungan Masyarakat', 'gig1_IT-Admin Jaringan/Sistem/Database', 'gig1_IT-Perangkat Keras', 'gig1_IT-Perangkat Lunak', 'gig1_Jurnalis/Editor', 'gig1_Keuangan / Investasi Perusahaan ', 'gig1_Kimia', 'gig1_Kontrol Proses', 'gig1_Lainnya/Kategori tidak tersedia', 'gig1_Layanan Pelanggan', 'gig1_Layanan Sosial/Konseling', 'gig1_Logistik/Rantai Pasokan', 'gig1_Makanan/Minuman/Pelayanan Restoran', 'gig1_Manufaktur', 'gig1_Mekanikal', 'gig1_Merchandising', 'gig1_Minyak/Gas', 'gig1_Pekerjaan Umum', 'gig1_Pelatihan & Pengembangan', 'gig1_Pemasaran/Pengembangan Bisnis', 'gig1_Pembelian/Manajemen Material', 'gig1_Pemeliharaan', 'gig1_Pendidikan', 'gig1_Penerbitan', 'gig1_Pengacara / Asisten Legal', 'gig1_Penjaminan Kualitas / QA', 'gig1_Penjualan - Jasa Keuangan', 'gig1_Penjualan - Korporasi', 'gig1_Penjualan - Teknik/Teknikal/IT', 'gig1_Penjualan Ritel', 'gig1_Perawatan Pribadi', 'gig1_Perbankan / Jasa Finansial ', 'gig1_Periklanan', 'gig1_Pertanian', 'gig1_Praktisi/Asisten Medis', 'gig1_Properti/Real Estate', 'gig1_Sains & Teknologi', 'gig1_Sekretaris', 'gig1_Seni / Desain Kreatif', 'gig1_Staf / Administrasi umum', 'gig1_Sumber Daya Manusia / HR', 'gig1_Survei Kuantitas', 'gig1_Teknik Elektro', 'gig1_Teknik Elektronika', 'gig1_Teknik Industri', 'gig1_Teknik Kimia', 'gig1_Teknik Lainnya', 'gig1_Teknik Lingkungan', 'gig1_Teknik Sipil/Konstruksi Bangunan', 'gig1_Teknikal & Bantuan Pelanggan', 'gig1_Teknologi Makanan/Ahli Gizi', 'gig1_Telesales/Telemarketing', 'gig1_Top Management / Manajemen Tingkat Atas', 'gig2_Akuntansi / Keuangan', 'gig2_Bangunan/Konstruksi', 'gig2_Hotel/Restoran', 'gig2_Komputer/Teknologi Informasi', 'gig2_Lainnya', 'gig2_Layanan Kesehatan', 'gig2_Manufaktur', 'gig2_Pelayanan', 'gig2_Pendidikan/Pelatihan', 'gig2_Penjualan / Pemasaran', 'gig2_Sains', 'gig2_Seni/Media/Komunikasi', 'gig2_Sumber Daya Manusia/Personalia', 'gig2_Teknik', 'industry_', 'industry_Agrikultural/Perkebunan/Peternakan Unggas/Perikanan', 'industry_Akunting / Audit / Layanan Pajak', 'industry_Asuransi', 'industry_Automobil/Mesin Tambahan Automotif/Kendaraan', 'industry_Bahan Kimia/Pupuk/Pestisida', 'industry_BioTeknologi/Farmasi/Riset klinik', 'industry_Call Center/IT-Enabled Services/BPO', 'industry_Elektrikal & Elektronik', 'industry_Hiburan/Media', 'industry_Hotel/Pariwisata', 'industry_Hukum/Legal', 'industry_Ilmu Pengetahuan & Teknologi', 'industry_Industri Berat/Mesin/Peralatan', 'industry_Jual Beli Saham/Sekuritas', 'industry_Jurnalisme', 'industry_Kayu/Fiber/Kertas', 'industry_Keamanan/Penegak hukum', 'industry_Kelautan/Aquakultur', 'industry_Kesehatan/Medis', 'industry_Komputer/Teknik Informatika (Perangkat Keras)', 'industry_Komputer/Teknik Informatika (Perangkat Lunak)', 'industry_Konstruksi/Bangunan/Teknik', 'industry_Konsultasi (Bisnis & Manajemen)', 'industry_Konsultasi (IT, Ilmu Pengetahuan, Teknis & Teknikal)', 'industry_Lainnya', 'industry_Layanan Umum/Tenaga Penggerak', 'industry_Lingkungan/Kesehatan/Keamanan', 'industry_Makanan & Minuman/Katering/Restoran', 'industry_Manajemen/Konsulting HR', 'industry_Manufaktur/Produksi', 'industry_Minyak/Gas/Petroleum', 'industry_Olahraga', 'industry_Organisasi Nirlaba/Pelayanan Sosial/LSM', 'industry_Pakaian', 'industry_Pameran/Manajemen acara/PIKP', 'industry_Pelayanan Arsitek/Desain Interior', 'industry_Pelayanan Perbaikan & Pemeliharaan', 'industry_Pemerintahan/Pertahanan', 'industry_Pendidikan', 'industry_Perawatan/Kecantikan/Fitnes', 'industry_Perbankan/Pelayanan Keuangan', 'industry_Percetakan/Penerbitan', 'industry_Periklanan/Marketing/Promosi/Hubungan Masyarakat', 'industry_Permata/Perhiasan', 'industry_Pertambangan', 'industry_Polymer/Plastik/Karet/Ban', 'industry_Produk Konsumen/Barang konsumen yang bergerak cepat', 'industry_Properti/Real Estate', 'industry_R&D', 'industry_Retail/Merchandise', 'industry_Seni/Desain/Fashion', 'industry_Tekstil/Garment', 'industry_Telekomunikasi', 'industry_Transportasi/Logistik', 'industry_Travel/Pariwisata', 'industry_Umum & Grosir', 'location_Aceh', 'location_Ambon', 'location_Badung', 'location_Bali', 'location_Balikpapan', 'location_Bandar Lampung', 'location_Bandung', 'location_Bangka', 'location_Bangka Belitung', 'location_Banjar', 'location_Banjarbaru', 'location_Banjarmasin', 'location_Banjarnegara', 'location_Banten', 'location_Bantul', 'location_Banyuwangi', 'location_Batam', 'location_Bekasi', 'location_Belitung', 'location_Bengkulu', 'location_Binjai', 'location_Bintan', 'location_Blitar', 'location_Bogor', 'location_Brebes', 'location_Bukittinggi', 'location_Cianjur', 'location_Cibinong', 'location_Cikarang', 'location_Cikupa', 'location_Cilacap', 'location_Cilegon', 'location_Cileungsi', 'location_Cimahi', 'location_Cirebon', 'location_Citeureup', 'location_Denpasar', 'location_Depok', 'location_Dumai', 'location_Gianyar', 'location_Gorontalo', 'location_Gowa', 'location_Gresik', 'location_Halmahera', 'location_Hulu Sungai Tengah', 'location_Jakarta Barat', 'location_Jakarta Pusat', 'location_Jakarta Raya', 'location_Jakarta Selatan', 'location_Jakarta Timur', 'location_Jakarta Utara', 'location_Jambi', 'location_Jawa Barat', 'location_Jawa Tengah', 'location_Jawa Timur', 'location_Jayapura', 'location_Jember', 'location_Jepara', 'location_Kalimantan Barat', 'location_Kalimantan Selatan', 'location_Kalimantan Tengah', 'location_Kalimantan Timur', 'location_Kalimantan Utara', 'location_Kapuas', 'location_Karawang', 'location_Kediri', 'location_Kendari', 'location_Kepulauan Riau', 'location_Kepulauan Seribu', 'location_Ketapang', 'location_Klaten', 'location_Klungkung', 'location_Kota Banda Aceh', 'location_Kotabaru', 'location_Kotawaringin Timur', 'location_Kudus', 'location_Kupang', 'location_Kuta', 'location_Kutai Timur', 'location_Lampung', 'location_Madiun', 'location_Madura', 'location_Magelang', 'location_Makassar', 'location_Malang', 'location_Maluku', 'location_Maluku Utara', 'location_Mamuju', 'location_Manado', 'location_Maros', 'location_Mataram', 'location_Medan', 'location_Metro', 'location_Minahasa', 'location_Mojokerto', 'location_Muara Enim', 'location_Nunukan', 'location_Nusa Tenggara Timur', 'location_Padang', 'location_Palangkaraya', 'location_Palembang', 'location_Palopo', 'location_Palu', 'location_Pandeglang', 'location_Pangkal Pinang', 'location_Papua', 'location_Pare-Pare', 'location_Pasuruan', 'location_Pekalongan', 'location_Pekanbaru', 'location_Pemalang', 'location_Penajam Paser Utara', 'location_Ponorogo', 'location_Pontianak', 'location_Poso', 'location_Prabumulih', 'location_Purwakarta', 'location_Purwokerto', 'location_Purworejo', 'location_Rangkasbitung', 'location_Riau', 'location_Salatiga', 'location_Samarinda', 'location_Semarang', 'location_Seminyak', 'location_Serang', 'location_Sibolga', 'location_Sidoarjo', 'location_Singkawang', 'location_Sleman', 'location_Sukabumi', 'location_Sulawesi Barat', 'location_Sulawesi Selatan', 'location_Sulawesi Tengah', 'location_Sulawesi Tenggara', 'location_Sulawesi Utara', 'location_Sumatera Barat', 'location_Sumatera Selatan', 'location_Sumatera Utara', 'location_Surabaya', 'location_Surakarta', 'location_Tanah Bumbu', 'location_Tangerang', 'location_Tanjung Balai', 'location_Tarakan', 'location_Tasikmalaya', 'location_Tegal', 'location_Ternate', 'location_Timika', 'location_Tuban', 'location_Ubud', 'location_Ungaran', 'location_Wonogiri ', 'location_Yogyakarta', 'position_CEO/GM/Direktur/Manajer Senior', 'position_Lulusan baru/Pengalaman kerja kurang dari 1 tahun', 'position_Manajer/Asisten Manajer', 'position_Pegawai (non-manajemen & non-supervisor)', 'position_Supervisor/Koordinator', 'senin - sabtu', 'size_', 'size_1000.0', 'size_200.0', 'size_2000.0', 'size_50.0', 'size_500.0', 'size_5000.0', 'turn_time_', 'turn_time_1.0', 'turn_time_10.0', 'turn_time_11.0', 'turn_time_12.0', 'turn_time_13.0', 'turn_time_14.0', 'turn_time_15.0', 'turn_time_16.0', 'turn_time_17.0', 'turn_time_18.0', 'turn_time_19.0', 'turn_time_2.0', 'turn_time_20.0', 'turn_time_21.0', 'turn_time_22.0', 'turn_time_23.0', 'turn_time_24.0', 'turn_time_25.0', 'turn_time_26.0', 'turn_time_27.0', 'turn_time_28.0', 'turn_time_29.0', 'turn_time_3.0', 'turn_time_30.0', 'turn_time_4.0', 'turn_time_5.0', 'turn_time_6.0', 'turn_time_7.0', 'turn_time_8.0', 'turn_time_9.0', 'type_Kontrak', 'type_Magang', 'type_Paruh Waktu', 'type_Penuh Waktu', 'type_Penuh Waktu, Kontrak', 'type_Penuh Waktu, Magang', 'type_Penuh Waktu, Paruh Waktu', 'type_Temporer', 'uniform']

original_cols = final_data.columns.tolist()

id = final_data['id']

missing_cols = set(train_cols) - set(final_data.columns.tolist())
for col in missing_cols:
    final_data[col] = 0

excess_cols = set(final_data.columns.tolist()) - set(train_cols)
for col in excess_cols:
    final_data.drop(col, axis=1, inplace=True)

sorted_cols = sorted(final_data.columns)
Final_X = final_data[sorted_cols]

print(Final_X.columns.tolist())
Final_X = np.array(Final_X)

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

Final_X = tf.convert_to_tensor(Final_X, dtype=tf.float64)
final_pred = model.predict(Final_X)

output_csv = 'output_lastditchh.csv'

with open(output_csv, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'salary'])
    for i in range(len(final_pred)):
        writer.writerow([id[i], int(final_pred[i][0])])

exit(0)
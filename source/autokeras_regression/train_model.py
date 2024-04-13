import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

data = pd.read_csv('../../library/optimized/full_dataset.csv')

data.fillna('', inplace=True)

# perks = data['perks'].str.split(',').explode().unique()
# 
# data['perks'] = data['perks'].str.split(',').apply(lambda x: [s.strip() for s in x])
# 
# unique_vals = set(val for sublist in data['perks'] for val in sublist)
# 
# for val in unique_vals:
#     data[val] = 0
# 
# for index, row in data.iterrows():
#     for val in row['perks']:
#         data.at[index, val] = 1
# 
# data.drop(columns=['perks'], inplace=True)






data = data.drop('id', axis=1)
data = pd.get_dummies(data)

sorted_cols = sorted(data.columns)
X = data[sorted_cols]
X = X.drop('salary', axis=1)
print(len(X.columns.tolist()))
# print(X)
X = np.asarray(X).astype('float64')
X = X.reshape(838240, 1)
# print(X)

y = data['salary']
# print(y)
y = np.asarray(y).astype('float64')
y = y.reshape(4960, 1)
print(max(y))
print(min(y))

print(max(y))
print(min(y))
# y = y / max(y)
print(y)
print(max(y))
print(min(y))

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# print(X_train)

print("importing tensorflow...")
import tensorflow as tf
print("tensorflow loaded. version: ", tf.__version__)

import autokeras as ak

loss_fn = tf.keras.losses.MeanSquaredError()


akmodel = ak.TextRegressor(
    overwrite=True,
    max_trials=1
)

import matplotlib.pyplot as plt
akmodel.fit(
    X, 
    y, 
    epochs=500, 
    batch_size=169,
    validation_split=0.2,
)

pred = akmodel.predict(X)

from sklearn.metrics import mean_squared_error
print(mean_squared_error(y, pred))

# train_loss = history.history['loss']
# val_loss = history.history['val_loss']
# 
# savename = 'b1024_l16_d169_r0-1_l1l2-2_2.png'
# 
# epochs = range(len(train_loss))
# plt.plot(epochs, train_loss, 'r', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.savefig(savename)
# plt.show()
# 
# akmodel.evaluate(X, y, verbose=2)
# akmodel.save('dnn_at_169feature-nodes.keras')
# 
# Complete_Data = pd.read_csv('../../library/optimized/fullest_dataset.csv')
# Complete_Data.fillna('', inplace=True)
# Complete_Data = pd.get_dummies(Complete_Data)
# complete_data_cols = ['e_d3', 'e_d4', 'e_dp', 'e_gp', 'e_none', 'e_s1', 'e_s2', 'e_s3', 'e_sma', 'e_smu/smk/stm', 'e_sp', 'experience_', 'experience_1.0', 'experience_10.0', 'experience_11.0', 'experience_12.0', 'experience_15.0', 'experience_17.0', 'experience_2.0', 'experience_20.0', 'experience_3.0', 'experience_4.0', 'experience_5.0', 'experience_6.0', 'experience_7.0', 'experience_8.0', 'gig1_Aktuaria/Statistik', 'gig1_Akuntansi Umum / Pembiayaan', 'gig1_Angkatan Bersenjata', 'gig1_Arsitek/Desain Interior', 'gig1_Audit & Pajak', 'gig1_Biomedis', 'gig1_Bioteknologi', 'gig1_Diagnosa/Lainnya', 'gig1_Digital Marketing', 'gig1_E-commerce', 'gig1_Farmasi', 'gig1_Geologi/Geofisika', 'gig1_Hiburan', 'gig1_Hotel/Pariwisata', 'gig1_Hubungan Masyarakat', 'gig1_IT-Admin Jaringan/Sistem/Database', 'gig1_IT-Perangkat Keras', 'gig1_IT-Perangkat Lunak', 'gig1_Jurnalis/Editor', 'gig1_Keuangan / Investasi Perusahaan ', 'gig1_Kimia', 'gig1_Kontrol Proses', 'gig1_Lainnya/Kategori tidak tersedia', 'gig1_Layanan Pelanggan', 'gig1_Layanan Sosial/Konseling', 'gig1_Logistik/Rantai Pasokan', 'gig1_Makanan/Minuman/Pelayanan Restoran', 'gig1_Manufaktur', 'gig1_Mekanikal', 'gig1_Merchandising', 'gig1_Minyak/Gas', 'gig1_Pekerjaan Umum', 'gig1_Pelatihan & Pengembangan', 'gig1_Pemasaran/Pengembangan Bisnis', 'gig1_Pembelian/Manajemen Material', 'gig1_Pemeliharaan', 'gig1_Pendidikan', 'gig1_Penerbitan', 'gig1_Pengacara / Asisten Legal', 'gig1_Penjaminan Kualitas / QA', 'gig1_Penjualan - Jasa Keuangan', 'gig1_Penjualan - Korporasi', 'gig1_Penjualan - Teknik/Teknikal/IT', 'gig1_Penjualan Ritel', 'gig1_Perawatan Pribadi', 'gig1_Perbankan / Jasa Finansial ', 'gig1_Periklanan', 'gig1_Pertanian', 'gig1_Praktisi/Asisten Medis', 'gig1_Properti/Real Estate', 'gig1_Sains & Teknologi', 'gig1_Sekretaris', 'gig1_Seni / Desain Kreatif', 'gig1_Staf / Administrasi umum', 'gig1_Sumber Daya Manusia / HR', 'gig1_Survei Kuantitas', 'gig1_Teknik Elektro', 'gig1_Teknik Elektronika', 'gig1_Teknik Industri', 'gig1_Teknik Kimia', 'gig1_Teknik Lainnya', 'gig1_Teknik Lingkungan', 'gig1_Teknik Sipil/Konstruksi Bangunan', 'gig1_Teknikal & Bantuan Pelanggan', 'gig1_Teknologi Makanan/Ahli Gizi', 'gig1_Telesales/Telemarketing', 'gig1_Top Management / Manajemen Tingkat Atas', 'gig2_Akuntansi / Keuangan', 'gig2_Bangunan/Konstruksi', 'gig2_Hotel/Restoran', 'gig2_Komputer/Teknologi Informasi', 'gig2_Lainnya', 'gig2_Layanan Kesehatan', 'gig2_Manufaktur', 'gig2_Pelayanan', 'gig2_Pendidikan/Pelatihan', 'gig2_Penjualan / Pemasaran', 'gig2_Sains', 'gig2_Seni/Media/Komunikasi', 'gig2_Sumber Daya Manusia/Personalia', 'gig2_Teknik', 'industry_', 'industry_Agrikultural/Perkebunan/Peternakan Unggas/Perikanan', 'industry_Akunting / Audit / Layanan Pajak', 'industry_Asuransi', 'industry_Automobil/Mesin Tambahan Automotif/Kendaraan', 'industry_Bahan Kimia/Pupuk/Pestisida', 'industry_BioTeknologi/Farmasi/Riset klinik', 'industry_Call Center/IT-Enabled Services/BPO', 'industry_Elektrikal & Elektronik', 'industry_Hiburan/Media', 'industry_Hotel/Pariwisata', 'industry_Hukum/Legal', 'industry_Ilmu Pengetahuan & Teknologi', 'industry_Industri Berat/Mesin/Peralatan', 'industry_Jual Beli Saham/Sekuritas', 'industry_Jurnalisme', 'industry_Kayu/Fiber/Kertas', 'industry_Keamanan/Penegak hukum', 'industry_Kelautan/Aquakultur', 'industry_Kesehatan/Medis', 'industry_Komputer/Teknik Informatika (Perangkat Keras)', 'industry_Komputer/Teknik Informatika (Perangkat Lunak)', 'industry_Konstruksi/Bangunan/Teknik', 'industry_Konsultasi (Bisnis & Manajemen)', 'industry_Konsultasi (IT, Ilmu Pengetahuan, Teknis & Teknikal)', 'industry_Lainnya', 'industry_Layanan Umum/Tenaga Penggerak', 'industry_Lingkungan/Kesehatan/Keamanan', 'industry_Makanan & Minuman/Katering/Restoran', 'industry_Manajemen/Konsulting HR', 'industry_Manufaktur/Produksi', 'industry_Minyak/Gas/Petroleum', 'industry_Olahraga', 'industry_Organisasi Nirlaba/Pelayanan Sosial/LSM', 'industry_Pakaian', 'industry_Pameran/Manajemen acara/PIKP', 'industry_Pelayanan Arsitek/Desain Interior', 'industry_Pelayanan Perbaikan & Pemeliharaan', 'industry_Pemerintahan/Pertahanan', 'industry_Pendidikan', 'industry_Perawatan/Kecantikan/Fitnes', 'industry_Perbankan/Pelayanan Keuangan', 'industry_Percetakan/Penerbitan', 'industry_Periklanan/Marketing/Promosi/Hubungan Masyarakat', 'industry_Permata/Perhiasan', 'industry_Pertambangan', 'industry_Polymer/Plastik/Karet/Ban', 'industry_Produk Konsumen/Barang konsumen yang bergerak cepat', 'industry_Properti/Real Estate', 'industry_R&D', 'industry_Retail/Merchandise', 'industry_Seni/Desain/Fashion', 'industry_Tekstil/Garment', 'industry_Telekomunikasi', 'industry_Transportasi/Logistik', 'industry_Travel/Pariwisata', 'industry_Umum & Grosir', 'position_CEO/GM/Direktur/Manajer Senior', 'position_Lulusan baru/Pengalaman kerja kurang dari 1 tahun', 'position_Manajer/Asisten Manajer', 'position_Pegawai (non-manajemen & non-supervisor)', 'position_Supervisor/Koordinator']
# 
# data_cols = data.columns.tolist()
# missing_cols = set(complete_data_cols) - set(data.columns.tolist())
# for col in missing_cols:
#     data[col] = 0
# 
# excess_cols = set(data.columns.tolist()) - set(complete_data_cols)
# for col in excess_cols:
#     data.drop(col, axis=1, inplace=True)
# sorted_cols = sorted(data.columns)
# Complete_X = data[sorted_cols]
# Complete_X = np.array(Complete_X)
# Complete_X = tf.convert_to_tensor(Complete_X, dtype=tf.float64)
# 
# prediction = akmodel.predict(Complete_X)
# print(prediction)
# 
# from sklearn.metrics import mean_squared_error
# print(mean_squared_error(y, prediction))
# 
exit(0)
import csv
import pandas as pd
import numpy as np

print("importing tensorflow...")
import tensorflow as tf
print("loaded tensorflow version: ", tf.__version__)

model = tf.keras.models.load_model('THE_MODEL_TO_END_ALL_MODELS.keras')

data = pd.read_csv('../../library/optimized/predict_dataset.csv')
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
data = pd.get_dummies(data)

train_cols = ['e_d3', 'e_d4', 'e_dp', 'e_gp', 'e_none', 'e_s1', 'e_s2', 'e_s3', 'e_sma', 'e_smu/smk/stm', 'e_sp', 'experience_', 'experience_1.0', 'experience_10.0', 'experience_11.0', 'experience_12.0', 'experience_15.0', 'experience_17.0', 'experience_2.0', 'experience_20.0', 'experience_3.0', 'experience_4.0', 'experience_5.0', 'experience_6.0', 'experience_7.0', 'experience_8.0', 'gig1_Aktuaria/Statistik', 'gig1_Akuntansi Umum / Pembiayaan', 'gig1_Angkatan Bersenjata', 'gig1_Arsitek/Desain Interior', 'gig1_Audit & Pajak', 'gig1_Biomedis', 'gig1_Bioteknologi', 'gig1_Diagnosa/Lainnya', 'gig1_Digital Marketing', 'gig1_E-commerce', 'gig1_Farmasi', 'gig1_Geologi/Geofisika', 'gig1_Hiburan', 'gig1_Hotel/Pariwisata', 'gig1_Hubungan Masyarakat', 'gig1_IT-Admin Jaringan/Sistem/Database', 'gig1_IT-Perangkat Keras', 'gig1_IT-Perangkat Lunak', 'gig1_Jurnalis/Editor', 'gig1_Keuangan / Investasi Perusahaan ', 'gig1_Kimia', 'gig1_Kontrol Proses', 'gig1_Lainnya/Kategori tidak tersedia', 'gig1_Layanan Pelanggan', 'gig1_Layanan Sosial/Konseling', 'gig1_Logistik/Rantai Pasokan', 'gig1_Makanan/Minuman/Pelayanan Restoran', 'gig1_Manufaktur', 'gig1_Mekanikal', 'gig1_Merchandising', 'gig1_Minyak/Gas', 'gig1_Pekerjaan Umum', 'gig1_Pelatihan & Pengembangan', 'gig1_Pemasaran/Pengembangan Bisnis', 'gig1_Pembelian/Manajemen Material', 'gig1_Pemeliharaan', 'gig1_Pendidikan', 'gig1_Penerbitan', 'gig1_Pengacara / Asisten Legal', 'gig1_Penjaminan Kualitas / QA', 'gig1_Penjualan - Jasa Keuangan', 'gig1_Penjualan - Korporasi', 'gig1_Penjualan - Teknik/Teknikal/IT', 'gig1_Penjualan Ritel', 'gig1_Perawatan Pribadi', 'gig1_Perbankan / Jasa Finansial ', 'gig1_Periklanan', 'gig1_Pertanian', 'gig1_Praktisi/Asisten Medis', 'gig1_Properti/Real Estate', 'gig1_Sains & Teknologi', 'gig1_Sekretaris', 'gig1_Seni / Desain Kreatif', 'gig1_Staf / Administrasi umum', 'gig1_Sumber Daya Manusia / HR', 'gig1_Survei Kuantitas', 'gig1_Teknik Elektro', 'gig1_Teknik Elektronika', 'gig1_Teknik Industri', 'gig1_Teknik Kimia', 'gig1_Teknik Lainnya', 'gig1_Teknik Lingkungan', 'gig1_Teknik Sipil/Konstruksi Bangunan', 'gig1_Teknikal & Bantuan Pelanggan', 'gig1_Teknologi Makanan/Ahli Gizi', 'gig1_Telesales/Telemarketing', 'gig1_Top Management / Manajemen Tingkat Atas', 'gig2_Akuntansi / Keuangan', 'gig2_Bangunan/Konstruksi', 'gig2_Hotel/Restoran', 'gig2_Komputer/Teknologi Informasi', 'gig2_Lainnya', 'gig2_Layanan Kesehatan', 'gig2_Manufaktur', 'gig2_Pelayanan', 'gig2_Pendidikan/Pelatihan', 'gig2_Penjualan / Pemasaran', 'gig2_Sains', 'gig2_Seni/Media/Komunikasi', 'gig2_Sumber Daya Manusia/Personalia', 'gig2_Teknik', 'industry_', 'industry_Agrikultural/Perkebunan/Peternakan Unggas/Perikanan', 'industry_Akunting / Audit / Layanan Pajak', 'industry_Asuransi', 'industry_Automobil/Mesin Tambahan Automotif/Kendaraan', 'industry_Bahan Kimia/Pupuk/Pestisida', 'industry_BioTeknologi/Farmasi/Riset klinik', 'industry_Call Center/IT-Enabled Services/BPO', 'industry_Elektrikal & Elektronik', 'industry_Hiburan/Media', 'industry_Hotel/Pariwisata', 'industry_Hukum/Legal', 'industry_Ilmu Pengetahuan & Teknologi', 'industry_Industri Berat/Mesin/Peralatan', 'industry_Jual Beli Saham/Sekuritas', 'industry_Jurnalisme', 'industry_Kayu/Fiber/Kertas', 'industry_Keamanan/Penegak hukum', 'industry_Kelautan/Aquakultur', 'industry_Kesehatan/Medis', 'industry_Komputer/Teknik Informatika (Perangkat Keras)', 'industry_Komputer/Teknik Informatika (Perangkat Lunak)', 'industry_Konstruksi/Bangunan/Teknik', 'industry_Konsultasi (Bisnis & Manajemen)', 'industry_Konsultasi (IT, Ilmu Pengetahuan, Teknis & Teknikal)', 'industry_Lainnya', 'industry_Layanan Umum/Tenaga Penggerak', 'industry_Lingkungan/Kesehatan/Keamanan', 'industry_Makanan & Minuman/Katering/Restoran', 'industry_Manajemen/Konsulting HR', 'industry_Manufaktur/Produksi', 'industry_Minyak/Gas/Petroleum', 'industry_Olahraga', 'industry_Organisasi Nirlaba/Pelayanan Sosial/LSM', 'industry_Pakaian', 'industry_Pameran/Manajemen acara/PIKP', 'industry_Pelayanan Arsitek/Desain Interior', 'industry_Pelayanan Perbaikan & Pemeliharaan', 'industry_Pemerintahan/Pertahanan', 'industry_Pendidikan', 'industry_Perawatan/Kecantikan/Fitnes', 'industry_Perbankan/Pelayanan Keuangan', 'industry_Percetakan/Penerbitan', 'industry_Periklanan/Marketing/Promosi/Hubungan Masyarakat', 'industry_Permata/Perhiasan', 'industry_Pertambangan', 'industry_Polymer/Plastik/Karet/Ban', 'industry_Produk Konsumen/Barang konsumen yang bergerak cepat', 'industry_Properti/Real Estate', 'industry_R&D', 'industry_Retail/Merchandise', 'industry_Seni/Desain/Fashion', 'industry_Tekstil/Garment', 'industry_Telekomunikasi', 'industry_Transportasi/Logistik', 'industry_Travel/Pariwisata', 'industry_Umum & Grosir', 'position_CEO/GM/Direktur/Manajer Senior', 'position_Lulusan baru/Pengalaman kerja kurang dari 1 tahun', 'position_Manajer/Asisten Manajer', 'position_Pegawai (non-manajemen & non-supervisor)', 'position_Supervisor/Koordinator']

# print(len(train_cols))
original_cols = data.columns.tolist()

id = data['id']

missing_cols = set(train_cols) - set(data.columns.tolist())
for col in missing_cols:
    data[col] = 0

excess_cols = set(data.columns.tolist()) - set(train_cols)
for col in excess_cols:
    data.drop(col, axis=1, inplace=True)

# print(data.columns.tolist())
# print(len(data.columns.tolist()))

# print(set(data.columns.tolist()) - set(train_cols))

sorted_cols = sorted(data.columns)
X = data[sorted_cols]
# X = X.drop(columns=['salary'])
print(X.columns.tolist())
X = np.array(X)

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

X = tf.convert_to_tensor(X, dtype=tf.float64)
inference = model.predict(X)
print(inference)
print(inference)

output_csv = 'output_maxima.csv'

with open(output_csv, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'salary'])
    for i in range(len(inference)):
        writer.writerow([id[i], int(inference[i][0])])
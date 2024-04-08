import csv
import os

def optimizePredict(predict_file, output_file):
    with open(predict_file, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

        Perk = {}
        for row in data:
            education = [
                ['Tidak' , 'e_none'],
                ['SMA'   , 'e_sma'],
                ['SMU'   , 'e_smu/smk/stm'],
                ['t P'   , 'e_sp'],
                ['D3'    , 'e_d3'],
                ['D4'    , 'e_d4'],
                ['S1'    , 'e_s1'],
                ['a P'   , 'e_dp'],
                ['r P'   , 'e_gp'],
                ['S2'    , 'e_s2'],
                ['S3'    , 'e_s3']
            ]

            # education = [
            #     ['Do'    , 'e_s3'],
            #     ['Ma'    , 'e_s2'],
            #     ['r P'   , 'e_gp'],
            #     ['a P'   , 'e_dp'],
            #     ['Sa'    , 'e_s1'],
            #     ['D4'    , 'e_d4'],
            #     ['D3'    , 'e_d3'],
            #     ['t P'   , 'e_sp'],
            #     ['SMU'   , 'e_smu/smk/stm'],
            #     ['SMA'   , 'e_sma'],
            #     ['Tidak' , 'e_none'],
            # ]

            ed = row[6]

            for i in range(0, len(education)):
                row.insert(7, 0)
 
            if 'Tidak terspesifikasi' in ed:
                row[7] = 1
            if 'SMA' in ed:
                row[8] = 1
            if 'SMU' in ed:
                row[9] = 1
            if 't P' in ed:
                row[10] = 1
            if 'D3' in ed:
                row[11] = 1
            if 'D4' in ed:
                row[12] = 1
            if 'Sarjana (S1)' in ed:
                row[13] = 1
            if 'a P' in ed:
                row[14] = 1
            if 'r P' in ed:
                row[15] = 1
            if 'S2' in ed:
                row[16] = 1
            if 'S3' in ed:
                row[17] = 1

            del row[1]
            del row[2]
            row[3] = row[3].split(' ')[0]
            del row[4]

            gigs = row[16].split(',')
            row[16] = gigs[0]
            row.insert(16, gigs[1])

            perks = row[18].split(';')
            for perk in perks:
                Perk[perk] = Perk.get(perk, 0) + 1

            row[19] = row[19].split(' ')[0]

            row[20] = row[20].split(' ')[(-2 if len(row[20].split(' ')) > 1 else -1)]

            del row[22]
            del row[19]
            
        UsedPerk =  {'': 1195, 'Asuransi kesehatan': 2497, 'Waktu regular, Senin - Jumat': 2461, 'Bisnis (contoh: Kemeja)': 1998, 'Tip': 1191, 'Parkir': 760, 'Penglihatan': 218, 'Tunjangan Pendidikan': 231, 'Kasual (contoh: Kaos)': 1000, 'Asuransi Gigi': 431, 'Pinjaman': 311, 'Olahraga (contoh: pusat kebugaran)': 225, 'Bonus': 37, 'Smart Casual Business': 23, '-': 68, 'Formil (contoh: Kemeja + Dasi)': 371, 'Mondays - Saturdays': 19, 'Others Benefits': 11, 'Senin-Sabtu': 42, 'Monday-Saturday': 33, 'BPJS': 70, 'Jam Bekerja yang Panjang': 106, 'uniform': 29, 'Tidy casual': 10, 'Uniform': 54, 'Monday - Saturday': 76, 'Seragam': 27, 'Senin - Sabtu': 60, 'Mondays-Saturdays': 19, 'Jamsostek': 21, 'BPJS Ketenagakerjaan': 20, 'Business, Uniform': 20, 'Formal Casual': 13, 'Smart Casual': 36, 'BPJS Kesehatan, BPJS Ketenagakerjaan': 23, 'BPJS Kesehatan dan BPJS Ketenagakerjaan': 17, 'BPJS, Bonus, Annual Reward': 13, 'Company Uniform': 12, 'Lunch BPJS Career Path': 13, 'Lunch and Private Insurance': 15, 'BPJS Kesehatan': 15, 'Other Benefits': 12, 'Scheduled performance appraisals': 15, 'Monday - Friday': 10, 'Transportasi': 12, 'Monday-Friday': 10, 'Monday to Saturday': 12, 'senin - sabtu': 11, 'Ketenagakerjaan': 10}

        for row in data:
            perks = row[18].split(';')
            
            final_perk = ""
            for perk in perks:
                if perk in UsedPerk:
                    final_perk += (perk)
                    final_perk += (',')
            
            row[18] = final_perk

            del row[1]
            del row[14]
            del row[16]
            del row[16]
                    
        print(UsedPerk)
        with open(output_file, 'w') as f:
            writer = csv.writer(f)

            # writer.writerow([
            #     'id', 
            #     'location', 
            #     'position', 
            #     'experience', 
            #     'education',
            #     'type', 
            #     'gig', 
            #     'perks', 
            #     'turn_time', 
            #     'size', 
            #     'industry', 
            #     'salary'
            # ])

            writer.writerow([
                'id', 
                # 'location', 
                'position', 
                'experience', 
                    'e_none',
                    'e_sma',
                    'e_smu/smk/stm',
                    'e_sp',
                    'e_d3',
                    'e_d4',
                    'e_s1',
                    'e_dp',
                    'e_gp',
                    'e_s2',
                    'e_s3',
                #'type', 
                'gig1', 
                'gig2',
                #'perks', 
                #'size', 
                'industry'
            ])
            writer.writerows(data)

if __name__ == '__main__':
    print(os.path.abspath(os.path.join(os.getcwd(), '../../')))

    predict_file = '../../library/predict_dataset.csv'
    output_file = '../../library/optimized/predict_dataset.csv'

    optimizePredict(predict_file, output_file)
    exit(0)
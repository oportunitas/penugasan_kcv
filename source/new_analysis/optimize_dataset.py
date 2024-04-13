import csv
import os

import numpy as np

def optimizeDataset(dataset_file, output_file):
    with open(dataset_file, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

        Perk = {}
        Location = {}
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

            Location[row[1]] = Location.get(row[1], 0) + 1

            perks = row[18].split(';')
            for perk in perks:
                Perk[perk] = Perk.get(perk, 0) + 1

            row[19] = row[19].split(' ')[0]

            row[20] = row[20].split(' ')[(-2 if len(row[20].split(' ')) > 1 else -1)]

            del row[22]
            # del row[19]
            
        UsedPerk =  {key: value for key, value in Perk.items() if value >= 10}
        UsedLocation = {key: value for key, value in Location.items() if value >= 0}

        for row in data:
            perks = row[18].split(';')
            
            final_perk = ""
            for perk in perks:
                if perk in UsedPerk:
                    final_perk += (perk)
                    final_perk += (',')
            
            row[18] = final_perk
            if row[1] not in UsedLocation:
                row[1] = 'Lainnya'

            # Remove columns
            #for i in range(0, 0):
            #    del row[1]
            #
            #for i in range(3, 23):
            #    del row[2]

            # row[22] = float(row[22])
            # row[17] = row[17] * 10000000

        # data = [
        #     row for row in data \
        #         if row[17] > 393004.75999999983 and \
        #         row[17] < 22500000.0
        # ]
            
                    
        print(UsedPerk)
        print(UsedLocation)

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
                'location', 
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
                'type', 
                'gig1', 
                'gig2',
                'perks', 
                'turn_time',
                'size', 
                'industry', 
                'salary'
            ])
            writer.writerows(data)

if __name__ == '__main__':
    print(os.path.abspath(os.path.join(os.getcwd(), '../../')))

    dataset_file = '../../library/predict_dataset.csv'
    output_file = '../../library/optimized/predict_dataset.csv'

    optimizeDataset(dataset_file, output_file)
    exit(0)
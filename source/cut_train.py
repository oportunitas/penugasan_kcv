import csv

input_file = '../library/raw/predict_case.csv'
output_file = '../library/predict_dataset.csv'



with open(input_file, 'r') as in_f, open(output_file, 'w') as out_f:
    reader = csv.reader(in_f, delimiter='|')
    writer = csv.writer(out_f)

    header = next(reader)

    for row in reader:
        if row[3] == 'USD':
            row[3] = 'IDR'
        print(row[5])
        writer.writerow(row)
import csv

def countOccurences(file, col_idx):
    counts = {}

    with open(file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            values = row[col_idx].split(',')
            for value in values:
               counts[value] = counts.get(value, 0) + 1
            # value = row[col_idx]
            # counts[value] = counts.get(value, 0) + 1
            
    sorted_counts = {
        k: v for k, v in sorted(
            counts.items(), key=lambda item: item[1]
        )
    }

    return sorted_counts

if __name__ == '__main__':
    csv_file = '../../library/optimized/full_dataset.csv'
    col_idx = 21
    occurences = countOccurences(csv_file, col_idx)

    for value, count in occurences.items():
        print(f'{value}: {count}')
    print(f'Total: {len(occurences)}')

    exit(0)
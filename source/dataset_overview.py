import pandas as pd

cut_train_file = '../library/cut_train.csv'
cut_train_data = pd.read_csv(cut_train_file, delimiter=',')

print(cut_train_data.columns)
tures = ['id', 'salary']
X = cut_train_data[tures]
print(X.describe())
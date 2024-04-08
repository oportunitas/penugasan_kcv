import pandas as pd
from sklearn.model_selection import train_test_split

full_dataset = pd.read_csv('../library/full_dataset.csv')

train_dataset, test_dataset = train_test_split(full_dataset, test_size=0.2)

train_dataset.to_csv('../library/train_dataset.csv', index=False)
test_dataset.to_csv('../library/test_dataset.csv', index=False)
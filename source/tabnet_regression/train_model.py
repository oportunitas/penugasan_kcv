import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

data = pd.read_csv('../../library/optimized/full_dataset.csv')

data.fillna('', inplace=True)
perks = data['perks'].str.split(',').explode().unique()

data['perks'] = data['perks'].str.split(',').apply(lambda x: [s.strip() for s in x])

unique_vals = set(val for sublist in data['perks'] for val in sublist)

for val in unique_vals:
    data[val] = 0

for index, row in data.iterrows():
    for val in row['perks']:
        data.at[index, val] = 1

data.drop(columns=['perks'], inplace=True)

# print(perks)

# for val in data['perks'].explode().unique():
#     data[val+'_flag'] = data['val'].apply(lambda x: 1 if val in x else 0)
# data = data.drop('perks', axis=1, inplace=True)

data = data.drop('id', axis=1)
data = pd.get_dummies(data)

test = pd.DataFrame(data, dtype='object', columns=[
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

# print(data.columns.tolist())
# print(data['salary'])

sorted_cols = sorted(data.columns)
X = data[sorted_cols]
X = X.drop('salary', axis=1)

X = np.array(X)
y = data['salary']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
y_train = y_train.values.reshape(-1, 1)
y_val = y_val.values.reshape(-1, 1)
# print(X_train)

from pytorch_tabnet.tab_model import TabNetRegressor
import torch

model = TabNetRegressor(
    optimizer_fn=torch.optim.Adam,
    mask_type='sparsemax'
)
model.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_val, y_val)],
    max_epochs=10000,
    patience=100,
    batch_size=1024,
)

inference = model.predict(X_val)
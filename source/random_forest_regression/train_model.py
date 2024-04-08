import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = pd.read_csv('../../library/optimized/full_dataset.csv')

data.fillna('', inplace=True)
perks = data['perks'].str.split(',').explode().unique()

data['perks'] = data['perks'].str.split(',').apply(lambda x: [s.strip() for s in x])

# Get all unique values from the split "val" column
unique_vals = set(val for sublist in data['perks'] for val in sublist)

# Create new columns for each unique value and initialize with zeros
for val in unique_vals:
    data[val] = 0

# Populate the new columns based on the presence of each element in the original "val" column
for index, row in data.iterrows():
    for val in row['perks']:
        data.at[index, val] = 1

# Drop the original "val" column
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
print(data['salary'])

X = data.drop('salary', axis=1)
y = data['salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# print(X_train)

mlp_regressor = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42, verbose=True)

# Fit the regressor to your training data
mlp_regressor.fit(X_train, y_train)

# Monitor the progress of the model
y_pred = mlp_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

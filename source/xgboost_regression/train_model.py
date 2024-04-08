import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

np.random.seed(42)

data = pd.read_csv('../../library/optimized/full_dataset.csv')

data.fillna('', inplace=True)
data = data.drop('id', axis=1)
data = pd.get_dummies(data)

sorted_cols = sorted(data.columns)
X = data[sorted_cols]
X = X.drop('salary', axis=1)
y = data['salary']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

import xgboost
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import joblib

# param_grid = {
#     'learning_rate': [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#     'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10],
#     'min_child_weight': [1, 3, 5, 7, 9],
#     'gamma': [0.0, 0.1, 0.2, 0.4, 0.8, 1.6],
#     'subsample': [0, 0.2, 0.4, 0.6, 0.8, 1.0],
#     'colsample_bytree': [0, 0.2, 0.4, 0.6, 0.8, 1.0],
#     'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
#     'lambda': [0.0, 0.1, 0.2, 0.4, 0.8, 1.6],
#     'alpha': [0.0, 0.1, 0.2, 0.4, 0.8, 1.6]
# }

model = XGBRegressor(eval_metric='rmse', random_state=42)

param_grid = {
    'n_estimators': [50],  
    'max_depth': [5],  
    'learning_rate': [0.225],  
    'min_child_weight': [7],  
    'subsample': [1.0],  
    'colsample_bytree': [0.4],  
    'gamma': [0],  
    'reg_alpha': [0.2],  
    'reg_lambda': [0, 0.1, 0.5],  
    'random_state': [42]
}

print("Training model...")
grid_search = GridSearchCV(
    estimator=model, 
    param_grid=param_grid,
    scoring='neg_mean_squared_error', 
    cv=5, 
    verbose=1,
)

print("Fitting model...")
grid_search.fit(
    X_train, y_train, 
    eval_set=[(X_val, y_val)], 
    verbose=1
)

print("Done fitting model.")
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

from sklearn.metrics import mean_squared_error

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_val)
# print(np.exp(y_pred))

best_model.save_model('xgboost_model.json')
print(mean_squared_error(y_val, y_pred))

results = grid_search.cv_results_
mean_val_losses = results['mean_test_score']
params = results['params']

import matplotlib.pyplot as plt

plt.figure(figsize=(16, 9))
plt.plot(range(len(mean_val_losses)), mean_val_losses, marker='o', linestyle='-')
plt.xlabel('Parameter Permutation')
plt.ylabel('Mean Validation Loss')
plt.title('Validation Loss for Each Parameter Permutation')
plt.xticks(range(len(mean_val_losses)), [str(param) for param in params], rotation=45, ha='right')
plt.grid(True)
plt.tight_layout()

plt.savefig('validation_loss_plot_all.png')

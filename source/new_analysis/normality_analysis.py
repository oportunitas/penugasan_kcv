import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm


data = pd.read_csv('../../library/optimized/full_dataset.csv')
plt.figure(figsize=(16, 9))
# data['salary'] = np.exp(data['salary'])
sns.histplot(data['salary'], kde=True, bins=50, color='blue', stat='density')

first_percentile = np.percentile(data['salary'], 0.03)
ninety_ninth_percentile = np.percentile(data['salary'], 98)
print(f'too poor: {first_percentile}, too rich: {ninety_ninth_percentile}')

plt.axvline(first_percentile, color='red', linestyle='--', label='1st percentile')
plt.axvline(ninety_ninth_percentile, color='green', linestyle='--', label='99th percentile')

plt.title('Salary Distribution')
plt.xlabel('Salary')
plt.ylabel('Density')
plt.show()
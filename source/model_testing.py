import pandas as pd
import numpy as np

print("importing tensorflow...")
import tensorflow as tf
print("loaded tensorflow version: ", tf.__version__)

model = tf.keras.models.load_model('salaries_min.keras')

data = pd.read_csv(
    '../library/full_dataset.csv',
    names=[
        'id',
        'job_title',
        'location',
        'salary_currency',
        'career_level',
        'experience_level',
        'education_level',
        'employment_type',
        'job_function',
        'job_benefits',
        'company_process_time',
        'company_size',
        'company_industry',
        'job_description',
        'salary'
    ]
)

data = np.array(data)
X = data[:, 1:13]
y = data[:, -1].astype(float)

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

label_encoders = []
for i in range(X.shape[1]):
    le = LabelEncoder()
    X[:, i] = le.fit_transform(X[:, i])
    label_encoders.append(le)

scaler = StandardScaler()
X = scaler.fit_transform(X)

loss_fn = tf.keras.losses.MeanSquaredError()
model.compile(
    loss=loss_fn,
    optimizer='adam'
)
inference = model.predict(X)

for i in range(len(inference)):
    print("Predicted\t:", inference[i], "\nActual\t\t: ", y[i], "\n")
    
model.evaluate(X, y, verbose=2)
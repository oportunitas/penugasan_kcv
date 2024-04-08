import pandas as pd
import numpy as np

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
print(X)
print(y)

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

label_encoders = []
for i in range(X.shape[1]):
    le = LabelEncoder()
    X[:, i] = le.fit_transform(X[:, i])
    label_encoders.append(le)

scaler = StandardScaler()
X = scaler.fit_transform(X)

print("importing tensorflow...")
import tensorflow as tf
print("tensorflow loaded. version: ", tf.__version__)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1)
])

loss_fn = tf.keras.losses.MeanSquaredError()

model.compile(
    loss=loss_fn,
    optimizer='adam'
)

import time
start_time = time.time()

class TimeLimitCallback(tf.keras.callbacks.Callback):
    def __init__(self, time_limit):
        super(TimeLimitCallback, self).__init__()
        self.time_limit = time_limit
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if time.time() - self.start_time >= self.time_limit:
            self.model.stop_training = True
            print(f'\nTime limit of {time.time() - self.start_time} seconds reached. Stopping training...')

model.fit(
    X, 
    y, 
    epochs=200000, 
    batch_size=1024, 
    validation_split=0.2, 
    callbacks=[
        TimeLimitCallback(60 * 60),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10000,
            verbose=1,
            restore_best_weights=True
        )
    ]
)

end_time = time.time()
print("Training time: ", end_time - start_time)

model.evaluate(X, y, verbose=2)
model.save('salaries_min.keras')

exit(0)

# print("dataset:")
# print(data)

# data_features = data.copy()
# print("features:")
# print(data_features)

# data_labels = data_features.pop('salary')
# print("labels:")
# print(data_labels)

# data_features = np.array(data_features)
# print("features array:")
# print(data_features)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    data_features, 
    data_labels, 
    test_size=0.2, 
    random_state=42
)


print("importing tensorflow...")
import tensorflow as tf
print("tensorflow loaded. version: ", tf.__version__)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(2048, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2048, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

loss_fn = tf.keras.losses.MeanSquaredError()

model.compile(
    loss=loss_fn,
    optimizer='adam'
)

model.fit(train_features, train_labels, epochs=50)
model.evaluate(test_features, test_labels, verbose=2)
model.save('mnist_model.keras')
import pandas as pd
import numpy as np

abalone_train = pd.read_csv(
    "https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv",
    names=[
        "Length", 
        "Diameter", 
        "Height", 
        "Whole weight", 
        "Shucked weight",
        "Viscera weight", 
        "Shell weight", 
        "Age"
    ]
)
print(abalone_train.head())

abalone_features = abalone_train.copy()
abalone_labels = abalone_features.pop("Age")

abalone_features = np.array(abalone_features)
print(abalone_features)
print(abalone_labels)

print("importing tensorflow...")
import tensorflow as tf
print("tensorflow loaded. version: ", tf.__version__)

abalone_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

loss_fn = tf.keras.losses.MeanSquaredError()

abalone_model.compile(
    loss=loss_fn,
    optimizer='adam'
)

abalone_model.fit(abalone_features, abalone_labels, epochs=20)
abalone_model.evaluate(abalone_features, abalone_labels, verbose=2)
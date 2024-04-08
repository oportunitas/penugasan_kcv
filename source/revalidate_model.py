import tensorflow as tf
import keras
print("TensorFlow version: ", tf.__version__)

model = keras.models.load_model('mnist_model.keras')

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
model.evaluate(x_test,  y_test, verbose=2)
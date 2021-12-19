import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28).astype("float32") / 255.0

print(x_train.shape)


model = keras.Sequential()
model.add(keras.Input(shape=(None, 28)))
# (None):represents the 28 timestamps;
# (28):represents pixels at each timestamp

model.add(layers.SimpleRNN(256, return_sequences=True, activation="relu"))
# (return_sequences=True): returns output at each timestamp, and therefore another Layer can be stacked upon

model.add(layers.SimpleRNN(256, activation="relu"))
# here only at the last timestamp, the output is passed to the next layer
model.add(layers.Dense(10))


model = keras.Sequential()
model.add(keras.Input(shape=(None, 28)))
model.add(layers.GRU(256, return_sequences=True, activation="tanh"))
model.add(layers.GRU(256, activation="tanh"))
model.add(layers.Dense(10))


model = keras.Sequential()
model.add(keras.Input(shape=(None, 28)))
model.add(layers.LSTM(256, return_sequences=True, activation="tanh"))
model.add(layers.LSTM(256, activation="tanh"))
model.add(layers.Dense(10))



model = keras.Sequential()
model.add(keras.Input(shape=(None, 28)))
model.add(
    layers.Bidirectional(
        layers.LSTM(256, return_sequences=True, activation="tanh")
    )
)
model.add(
    layers.Bidirectional(
        layers.LSTM(256, activation="tanh")
    )
)
model.add(layers.Dense(10))








print(model.summary())

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=['accuracy']
)

model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)
model.evaluate(x_test, y_test, batch_size=64, verbose=2)

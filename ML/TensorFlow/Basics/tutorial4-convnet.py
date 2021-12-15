import os
os.environ['TF_CPP_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import mnist

physical_device = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_device[0], True)


#(x_train, y_train), (x_test, y_test) = cifar10.load_data()
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


input_shape = x_train.shape[1:]

if len(input_shape) == 2:
    input_shape = input_shape + (1, )


model = keras.Sequential(
    [
        keras.Input(shape=input_shape), # (dim1, dim2, channels)
        layers.Conv2D(32,  # channels/filters
                      3,   # (3, 3) kernels size
                      padding='valid',
                      activation='relu',
                      ),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10),
    ]
)


def my_model():
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(32, 3)(inputs)
    x = layers.BatchNormalization()(x)  # ... is but inbetween Conv2D layer its activation, therefore is split
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3)(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)

    x = layers.Flatten()(x)

    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


model = my_model()



print(model.summary())

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # because no softmax as activation in last layer
    optimizer=keras.optimizers.Adam(lr=3e-4),
    metrics=['accuracy']
)

model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)
model.evaluate(x_test, y_test, batch_size=64, verbose=2)


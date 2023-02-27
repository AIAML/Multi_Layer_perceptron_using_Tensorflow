import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()

print('x_train_full shape: ', x_train_full.shape, 'x_train_full type: ', x_train_full.dtype)
print('y_train_full shape: ', y_train_full.shape, 'y_train_full type: ', y_train_full.dtype)
print('x_test shape: ', x_test.shape, 'x_test type: ', x_test.dtype)
print('y_test shape:', y_test.shape, 'y_test type', y_test.dtype)


x_valid, x_train = x_train_full[50000:] / 255.0, x_train_full[:50000] / 255.0
y_valid, y_train = y_train_full[50000:], y_train_full[:50000]

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))
model.summary()
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
history = model.fit(x_train, y_train, epochs=30, validation_data=(x_valid, y_valid))


pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
plt.show()
plt.savefig("d:/t.jpg")
import tensorflow as tf
from tensorflow.keras import layers, models

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train.astype("float32") / 255, x_test.astype("float32") / 255

model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Conv2D(32, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),
    layers.Conv2D(96, 3, activation="relu", padding="same"),
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.25),
    layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=64, epochs=8, validation_split=0.1)
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {accuracy:.1%} (loss {loss:.3f})")
model.save("cifar10.keras")

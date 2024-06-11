import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense

import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


class CNNmodel:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.create_model()

    def create_model(self):
        model = Sequential()
        model.add(
            Conv2D(
                32,
                kernel_size=9,
                padding="same",
                activation="relu",
                input_shape=(32, 32, 3),
            )
        )
        model.add(MaxPool2D(pool_size=2))

        model.add(Conv2D(64, kernel_size=7, padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=2))

        model.add(Conv2D(128, kernel_size=3, padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=2))

        model.add(Flatten())
        model.add(Dense(500, activation="relu"))
        model.add(Dense(8, activation="softmax"))
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        return model

    def train(self, train_data, batch_size, epochs):
        X = np.array([img.data for img in train_data.images], dtype=np.float32)
        y = [img.label for img in train_data.images]

        # Fit the label encoder with all possible labels
        self.le = LabelEncoder()
        self.le.fit(np.unique(y))
        y = self.le.transform(y)
        # substract 1 to the labels to have them from 0 to 7
        y = y - 1
        y = to_categorical(y, num_classes=self.num_classes)

        self.model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=1)
        print("Model trained with CNN")

    def evaluate(self, val_data):
        X = []
        y = []

        for img in val_data.images:
            if img.label in self.le.classes_:
                X.append(img.data)
                y.append(img.label)

        X = np.array(X, dtype=np.float32)

        y = self.le.transform(y)
        y = y - 1

        y = to_categorical(y, num_classes=self.num_classes)

        return self.model.evaluate(X, y)

    def predict(self, x):
        return self.model.predict(x)

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model = load_model(filename)

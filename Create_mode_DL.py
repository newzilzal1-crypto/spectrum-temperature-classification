import os

from keras import Input, Sequential
from keras.layers import LSTM, Activation, BatchNormalization, Conv1D, Dense, Dropout, Flatten, MaxPooling1D

# All builders take `num_class` as the list of class names and save the (untrained) model to model/<titel>.keras.
# They are compiled for integer labels; use loss='categorical_crossentropy' instead if you pass one-hot labels.
LOSS = "sparse_categorical_crossentropy"


def _finish(model, titel):
    model.compile(optimizer="adam", loss=LOSS, metrics=["accuracy"])
    model.summary()
    os.makedirs("model", exist_ok=True)
    model.save(os.path.join("model", f"{titel}.keras"))
    return model


def mode1(x, num_class, titel):
    model = Sequential()
    model.add(Input(shape=(x.shape[1], 1)))
    model.add(Conv1D(512, 1))
    model.add(Activation("relu"))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(256, 1))
    model.add(Activation("relu"))
    model.add(MaxPooling1D(2))
    model.add(LSTM(512))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.15))
    model.add(Dense(2048, activation="relu"))
    model.add(Dense(1024, activation="relu"))
    model.add(Dense(len(num_class), activation="softmax"))
    return _finish(model, titel)


def modeCNN_LSTM(x, num_class, titel):
    # Same architecture as mode1; previously compiled with binary_crossentropy, which is wrong for >2 classes.
    return mode1(x, num_class, titel)


def modeCNN(x, num_class, titel):
    model = Sequential()
    model.add(Input(shape=(x.shape[1], 1)))
    model.add(Conv1D(512, 1))
    model.add(Activation("relu"))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(256, 1))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.15))
    model.add(Dense(2048, activation="relu"))
    model.add(Dense(1024, activation="relu"))
    model.add(Dense(len(num_class), activation="softmax"))
    return _finish(model, titel)


def modeRNN(X_train, num_class, titel):
    regressor = Sequential()
    regressor.add(Input(shape=(X_train.shape[1], 1)))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.2))
    # Softmax output so the network produces class probabilities (it had no activation before).
    regressor.add(Dense(units=len(num_class), activation="softmax"))
    return _finish(regressor, titel)

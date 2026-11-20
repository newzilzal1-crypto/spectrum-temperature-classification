from keras import Input, Sequential
from keras.layers import LSTM, Activation, BatchNormalization, Conv1D, Dense, Dropout, Flatten, MaxPooling1D
from keras.losses import CategoricalCrossentropy

from utilities_DL import f1_m, precision_m, recall_m

epochs = 100
batch_size = 64
learning_rate = 1e-3
random_state = 42
train_ratio = 0.70
validation_ratio = 0.20
test_ratio = 0.10
show = False  # True opens every plot in a window and pauses the script until it is closed

file_csv = "Data.xlsx"
path_folder = "Data"

# 'temperature' maps one-to-one onto 'Class' in Data.xlsx (0-20 -> Class1, 40 -> Class2,
# 60 -> Class3, 80-100 -> Class4). Keeping it lets a model reach ~100% without using the spectrum.
drop_features = ["temperature"]

optimizer_to_legend = {'Adam optimization': 'Adam optimization'}
list_metric = ['loss', 'accuracy', 'f1_m', 'precision_m', 'recall_m']

metrics = ["accuracy", f1_m, precision_m, recall_m]


def create_network_CNN(X_train, num_class, opt):
    mode = Sequential()
    mode.add(Input(shape=(X_train.shape[1], 1)))
    mode.add(Conv1D(512, 1))
    mode.add(Activation("relu"))
    mode.add(MaxPooling1D(2))
    mode.add(Conv1D(256, 1))
    mode.add(Activation("relu"))
    mode.add(MaxPooling1D(2))
    mode.add(Dropout(0.25))
    mode.add(BatchNormalization())
    mode.add(Flatten())
    mode.add(Dense(1024, activation="relu"))
    mode.add(Dense(num_class, activation="softmax"))
    mode.compile(loss=CategoricalCrossentropy(), optimizer=opt, metrics=metrics)
    mode.summary()
    return mode


def modeCNN_LSTM(X_train, num_class, opt):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], 1)))
    model.add(Conv1D(512, 1))
    model.add(Activation("relu"))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(256, 1))
    model.add(Activation("relu"))
    model.add(MaxPooling1D(2))
    model.add(LSTM(128))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.15))
    model.add(Dense(1024, activation="relu"))
    model.add(Dense(num_class, activation="softmax"))
    model.compile(loss=CategoricalCrossentropy(), optimizer=opt, metrics=metrics)
    model.summary()
    return model

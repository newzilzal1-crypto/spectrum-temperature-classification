import os

# Cap the math-library thread pools BEFORE numpy/scipy load. On machines with many cores each bundled
# OpenBLAS copy reserves per-thread buffers; when Windows' commit limit is nearly used up this fails
# ("OpenBLAS error: Memory allocation") and corrupts memory (seen as an XGBoost "access violation").
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "4")


import pandas as pd
import tensorflow as tf
from keras.optimizers import Adam

from init_paramaters import (batch_size, drop_features, epochs, file_csv, learning_rate, list_metric,
                             modeCNN_LSTM, optimizer_to_legend, path_folder, random_state, show, test_ratio,
                             train_ratio, validation_ratio)
from utilities_DL import (check_system, create_all_folder, peprossing, save_history, save_model, scale_splits,
                          show_evalution, split_data_)

# This script trains the CNN-LSTM network (a recurrent layer after two conv blocks).
type_CLass = "System_RNN"
model_name = "CNN_LSTM"

root = f"Case_{os.path.basename(file_csv).split('.')[0]}"
data = pd.read_excel(os.path.join(path_folder, file_csv))
data, X, y, num_class, class_names, label_encoder = peprossing(data, drop_features)
X_train, y_train, x_val, y_val, X_test, y_test = split_data_(X, y, train_ratio, validation_ratio, test_ratio,
                                                             random_state)
X_train, x_val, X_test, scaler = scale_splits(X_train, x_val, X_test)
tf.random.set_seed(random_state)

for tite in optimizer_to_legend:
    base_optimizer = Adam(learning_rate=learning_rate)  # a fresh optimizer for every model
    path = os.path.join(type_CLass, root, tite)
    create_all_folder(path)
    model_1 = modeCNN_LSTM(X_train, num_class, base_optimizer)

    history = model_1.fit(X_train, tf.one_hot(y_train, depth=num_class),
                          validation_data=(x_val, tf.one_hot(y_val, depth=num_class)),
                          batch_size=batch_size, epochs=epochs, verbose=1)
    save_history(path, history)
    save_model(path, model_1, model_name)
    show_evalution(path, tite, history, list_metric, show, root)
    check_system(path, X_train, y_train, "train", model_1, show, root, tite, class_names, label_encoder)
    check_system(path, X_test, y_test, "test", model_1, show, root, tite, class_names, label_encoder)

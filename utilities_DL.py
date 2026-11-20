import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import backend as K
from keras import ops
from pycm import ConfusionMatrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

class_param = ["AUC", "FPR", "PPV", "TP", "TN", "FP", "FN", "F1", 'MCC', 'TPR Micro', 'ACC', 'PPV Macro', 'ERR',
               'PPV Micro', 'Overall ACC']


# Batch-wise approximations used only for the training curves.
# The final, exact metrics are computed with pycm in check_system().
# Keras 3 removed K.sum / K.round / K.clip, so keras.ops is used instead.
def recall_m(y_true, y_pred):
    true_positives = ops.sum(ops.round(ops.clip(y_true * y_pred, 0, 1)))
    possible_positives = ops.sum(ops.round(ops.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def precision_m(y_true, y_pred):
    true_positives = ops.sum(ops.round(ops.clip(y_true * y_pred, 0, 1)))
    predicted_positives = ops.sum(ops.round(ops.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def peprossing(data, drop_features=()):
    """Encode labels and drop leaking columns. Scaling is done AFTER the split (see scale_splits)."""
    X = data.drop(['Class', *[c for c in drop_features if c in data.columns]], axis=1)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data['Class'])
    class_names = list(label_encoder.classes_)  # same order as the encoded labels
    return data, X.to_numpy(dtype="float32"), y, len(class_names), class_names, label_encoder


def scale_splits(x_train, *others):
    """Fit the scaler on the training data only, then apply it to the other splits."""
    scaler = StandardScaler().fit(x_train)
    return (scaler.transform(x_train), *[scaler.transform(x) for x in others], scaler)


def create_all_folder(name_data):
    for sub in ["metrics", "history", "model", "Confusion Matrix", "multi_classification",
                "summarized_classification"]:
        os.makedirs(os.path.join(name_data, sub), exist_ok=True)


def split_data(X, y, test_size, random_state=0):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def split_data_(X, y, train_ratio, validation_ratio, test_ratio, random_state=42):
    """Stratified, reproducible train / validation / test split.

    Returns x_train, y_train, x_val, y_val, x_test, y_test.
    """
    x_train, x_rest, y_train, y_rest = train_test_split(X, y, test_size=1 - train_ratio, stratify=y,
                                                        random_state=random_state)
    x_val, x_test, y_val, y_test = train_test_split(x_rest, y_rest,
                                                    test_size=test_ratio / (test_ratio + validation_ratio),
                                                    stratify=y_rest, random_state=random_state)
    return x_train, y_train, x_val, y_val, x_test, y_test


def check_system(name_data, X, Y, type_che, model_1, show, root, tite, class_name, label_encoder):
    pred_class = np.argmax(model_1.predict(X.astype('float32'), verbose=0), axis=1)
    true_value = label_encoder.inverse_transform(Y)
    pred_class = label_encoder.inverse_transform(pred_class)
    cm = ConfusionMatrix(actual_vector=list(true_value), predict_vector=list(pred_class))
    cm.stat(summary=True)

    cm.save_csv(os.path.join(name_data, "multi_classification", f"multi_classification_report_{type_che}"),
                class_param=class_param, header=True)
    cm.save_stat(os.path.join(name_data, "summarized_classification",
                              f"summarized_classification_report_{type_che}"), class_param=class_param)
    cm.plot(cmap=plt.cm.Reds, number_label=True, plot_lib="matplotlib", class_name=class_name,
            title=f"{root} ({type_che})\nACC=[{cm.ACC_Macro:.4f}]")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(name_data, "Confusion Matrix", f"Confusion Matrix-{type_che}.png"))
    if show:
        plt.show()
    plt.close()


def show_evalution(name_data, tite, history, listname, show, root):
    for name in listname:
        acc_list = [s for s in history.history.keys() if name in s and 'val' not in s]
        val_acc_list = [s for s in history.history.keys() if name in s and 'val' in s]
        if not acc_list:
            continue
        plt.figure(figsize=(8, 4))
        epochs = range(1, len(history.history[acc_list[0]]) + 1)
        for l_acc, l_val_acc in zip(acc_list, val_acc_list):
            plt.plot(epochs, history.history[l_acc], 'b',
                     label=f'Training {name} ({history.history[l_acc][-1]:.5f})')
            plt.plot(epochs, history.history[l_val_acc], 'g',
                     label=f'Validation {name} ({history.history[l_val_acc][-1]:.5f})')
        plt.title(f'Training and validation {name}')
        plt.xlabel("Epochs")
        plt.ylabel(name)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(name_data, "metrics", f"{name}.png"))
        if show:
            plt.show()
        plt.close()


def save_history(root, history):
    pd.DataFrame(history.history).to_csv(os.path.join(root, "history", "history.csv"))


def save_model(root, model_1, name="CNN"):
    model_1.save(os.path.join(root, "model", f"{name}.keras"))

import os

# Cap the math-library thread pools BEFORE numpy/scipy load. On machines with many cores each bundled
# OpenBLAS copy reserves per-thread buffers; when Windows' commit limit is nearly used up this fails
# ("OpenBLAS error: Memory allocation") and corrupts memory (seen as an XGBoost "access violation").
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "4")


import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utilities_ML import create_folder, do_dynmic

# 'temperature' maps one-to-one onto 'Class' (0-20 -> Class1, 40 -> Class2, 60 -> Class3, 80-100 -> Class4),
# so it is removed to stop the models from simply reading the label off it.
DROP_FEATURES = ["temperature"]


def get_dataset(x_Class):
    df = pd.read_excel(os.path.join("Data", "Data.xlsx"))
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[x_Class])
    X = df.drop(columns=[x_Class, *[c for c in DROP_FEATURES if c in df.columns]])
    return X, y, list(label_encoder.classes_)


def data_set1(type_, cv, xtidata, test_size, x_Class):
    X, y, labels = get_dataset(x_Class)
    print("Features:", list(X.columns))
    print("Classes:", labels)
    # Outlier clipping and scaling happen inside do_classfy, after the train/test split.
    return do_dynmic(type_, cv, X, y, xtidata, test_size, labels)


def do_System(ch):
    type_ = "System_0"
    create_folder(type_, ["data"],
                  ["multi_classification_report", "confusion_matrix",
                   "Remove_Outliers",
                   "summarized_classification_report"])
    data_set1(type_, cv, "data", test_size, "Class")


if __name__ == '__main__':
    test_size = 0.30
    cv = 3
    os.makedirs("System_0", exist_ok=True)
    do_System(0)

import os
from pathlib import Path
from shutil import rmtree

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report, confusion_matrix, f1_score,
                             multilabel_confusion_matrix, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from xgboost import XGBClassifier

# Default class names; do_classfy() replaces them with the real names from the label encoder.
display_labels = ["Class1", "Class2", "Class3", "Class4"]


def _report(TN, FP, FN, TP):
    TPR = TP / (TP + FN) if (TP + FN) != 0 else 0
    TNR = TN / (TN + FP) if (TN + FP) != 0 else 0
    PPV = TP / (TP + FP) if (TP + FP) != 0 else 0
    report = {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
              'TPR': TPR, 'Recall': TPR, 'Sensitivity': TPR,
              'TNR': TNR, 'Specificity': TNR,
              'FPR': FP / (FP + TN) if (FP + TN) != 0 else 0,
              'FNR': FN / (FN + TP) if (FN + TP) != 0 else 0,
              'PPV': PPV, 'Precision': PPV,
              'F1 Score': 2 * (PPV * TPR) / (PPV + TPR) if (PPV + TPR) != 0 else 0
              }
    return report


def multi_classification_report(y_test, y_pred, labels=None, encoded_labels=True, as_frame=False):
    conf_labels = None if encoded_labels else labels
    conf_mat = multilabel_confusion_matrix(y_test, y_pred, labels=conf_labels)
    report = dict()
    counter = np.arange(len(conf_mat)) if labels is None else labels
    for i, name in enumerate(counter):
        TN, FP, FN, TP = conf_mat[i].ravel()
        report[name] = _report(TN, FP, FN, TP)

    if as_frame:
        return pd.DataFrame(report)
    return report


def summarized_classification_report(y_true, y_pred, as_frame=False):
    report = dict()
    for avg in ['micro', 'macro', 'weighted']:
        report[avg] = {'Precision': precision_score(y_true, y_pred, average=avg, zero_division=0),
                       'Recall': recall_score(y_true, y_pred, average=avg, zero_division=0),
                       'F1 Score': f1_score(y_true, y_pred, average=avg, zero_division=0)}
    if as_frame:
        return pd.DataFrame(report)
    return report


def Handling_Outlier(type_, xtidata, df_, show=False, max_plots=3):
    """Clip every column to [Q1 - 1.5*IQR, Q3 + 1.5*IQR].

    Call this on the TRAINING features only. Returns (clipped copy, lower bounds, upper bounds),
    so the same bounds can be applied to the test set without leaking test information.
    """
    original = df_.copy()
    clipped = df_.copy()
    q1, q3 = df_.quantile(0.25), df_.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    out_dir = os.path.join(type_, "Remove_Outliers", xtidata)
    with open(os.path.join(out_dir, "outlier.csv"), "w") as csv:
        csv.write("columns,Q1,Q3,IQR\n")
        for col in df_.columns:
            csv.write(f"{col},{q1[col]},{q3[col]},{iqr[col]}\n")
    clipped = clipped.clip(lower, upper, axis=1)
    clipped.describe().T.to_csv(os.path.join(out_dir, "outlier_description.csv"))

    if show:
        for col in tqdm(list(df_.columns)[:max_plots]):
            fig = plt.figure()
            plt.subplot(2, 2, 1)
            sns.histplot(original[col], label=col)
            plt.subplot(2, 2, 2)
            sns.boxplot(x=original[col], color="#66c2a5")
            plt.subplot(2, 2, 3)
            sns.histplot(clipped[col], label=col)
            plt.subplot(2, 2, 4)
            sns.boxplot(x=clipped[col], color="#66c2a5")
            fig.suptitle(f"Detect and Remove Outliers of Feature\n[{col}]", x=0.5, y=0.93, fontsize=12,
                         fontweight="bold")
            plt.tight_layout()
            plt.subplots_adjust(top=0.8)
            plt.savefig(os.path.join(out_dir, f"Feature[{col.replace('/', '_')}].png"))
            plt.close()
    return clipped, lower, upper


def do_classfy(type_, cv, X, y, xftitel, xtidata, test_size, labels=None, random_state=42):
    global display_labels
    if labels is not None:
        display_labels = list(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y,
                                                        random_state=random_state)
    # Outlier bounds and scaler are learned from the training split only.
    X_train, lower, upper = Handling_Outlier(type_, xtidata, X_train, True)
    X_test = X_test.clip(lower, upper, axis=1)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print('X_train shape = ', X_train.shape)
    print('y_train shape = ', y_train.shape)
    print('X_test shape = ', X_test.shape)
    print('y_test shape = ', y_test.shape)

    return {
        "XGBClassifier": Baseline_Model_xgb(type_, cv, X_train, y_train, X_test, y_test, xftitel, xtidata),
        "RandomForest": Baseline_Model_RandomForest(type_, cv, X_train, y_train, X_test, y_test, xftitel, xtidata),
        "GaussianNB": Baseline_Model_GaussianNB(type_, cv, X_train, y_train, X_test, y_test, xftitel, xtidata),
        "DecisionTreeClassifier": Baseline_Model_DecisionTreeClassifier(type_, cv, X_train, y_train, X_test,
                                                                        y_test, xftitel, xtidata),
        "SVC": Baseline_Model_svm(type_, cv, X_train, y_train, X_test, y_test, xftitel, xtidata),
        "MLP": Baseline_MLPClassifier(type_, cv, X_train, y_train, X_test, y_test, xftitel, xtidata),
    }


def dete_result(parh):
    for path in Path(parh).glob("**/*"):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            rmtree(path)


def create_folder(type_, xlist, xxlist):
    for item in xlist:
        for xx in xxlist:
            os.makedirs(os.path.join(type_, xx, item), exist_ok=True)
            dete_result(os.path.join(type_, xx, item))


def visuall_Missing_Data(type_, df, xtidata, type):
    Missing_Data = {col: [df[col].isnull().sum(), f'% {np.round(np.mean(df[col].isnull() * 100), 3)}'] for col in
                    df.columns
                    if df[col].isnull().any()}
    pd.DataFrame.from_dict(Missing_Data).to_csv(f"{type_}//Missing Data//{xtidata}//{type}.csv")
    plt.figure(figsize=(18, 8))
    colours = ['#34495E', 'seagreen']
    sns.heatmap(df.isnull(), cmap=sns.color_palette(colours))
    plt.suptitle(f"{xtidata}")
    plt.savefig(f"{type_}//Missing Data//{xtidata}//show_Missing_Data_{type}.png")
    plt.close()
    sns.displot(data=df.isnull().melt(value_name='missing'), y='variable', hue='missing', multiple='fill', height=8,
                aspect=1.1)
    plt.suptitle(f"{xtidata}")
    plt.axvline(0.4, color='r')
    # Saved under a different name so it no longer overwrites the heatmap above.
    plt.savefig(f"{type_}//Missing Data//{xtidata}//show_Missing_Data_ratio_{type}.png")
    plt.close()


def do_dynmic(type_, cv, X, y, xtidata, test_size, labels=None):
    return do_classfy(type_, cv, X, y, "Features aLL", xtidata, test_size, labels)


def princ(type_, xtidata, xftitel, xxtt, y_r, y_pred):
    A = multi_classification_report(y_r, y_pred, labels=display_labels, encoded_labels=True, as_frame=True)
    A.to_csv(f'{type_}//multi_classification_report//{xtidata}//{xxtt}-{xftitel}.csv')
    B = summarized_classification_report(y_r, y_pred, as_frame=True)
    B.to_csv(f'{type_}//summarized_classification_report//{xtidata}//{xxtt}-{xftitel}.csv')


def _save_confusion_matrix(type_, xtidata, title, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(display_labels)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    fig, ax = plt.subplots()
    plt.suptitle(title)
    disp.plot(ax=ax)
    plt.savefig(f'{type_}//confusion_matrix//{xtidata}//{title}.png')
    plt.close(fig)


def _baseline_model_report(clf, name, type_, X_tr, y_tr, X_te, y_te, xftitel, xtidata):
    """Fit once, write reports + confusion matrices for train and test, return test probabilities."""
    clf.fit(X_tr, y_tr)
    print(f'Data {name} Classification Report:\n')
    for split, X_, y_ in (("Training", X_tr, y_tr), ("Testing", X_te, y_te)):
        pred = clf.predict(X_)
        print(f'{split} Data:\n', classification_report(y_, pred, zero_division=0))
        princ(type_, xtidata, f"{xftitel}_{split}", name, y_, pred)
        _save_confusion_matrix(type_, xtidata, f"{split} Data {name}-{xftitel}", y_, pred)
    return clf.predict_proba(X_te)


def Baseline_Model_xgb(type_, cv, X_train, y_train, X_test, y_test, xftitel, xtidata):
    return _baseline_model_report(XGBClassifier(), "XGBClassifier", type_, X_train, y_train, X_test, y_test,
                                  xftitel, xtidata)


def Baseline_Model_RandomForest(type_, cv, X_train, y_train, X_test, y_test, xftitel, xtidata):
    return _baseline_model_report(RandomForestClassifier(), "RandomForest", type_, X_train, y_train, X_test,
                                  y_test, xftitel, xtidata)


def Baseline_Model_GaussianNB(type_, cv, X_train, y_train, X_test, y_test, xftitel, xtidata):
    return _baseline_model_report(GaussianNB(), "GaussianNB", type_, X_train, y_train, X_test, y_test,
                                  xftitel, xtidata)


def Baseline_Model_DecisionTreeClassifier(type_, cv, X_train, y_train, X_test, y_test, xftitel, xtidata):
    return _baseline_model_report(DecisionTreeClassifier(), "DecisionTreeClassifier", type_, X_train, y_train,
                                  X_test, y_test, xftitel, xtidata)


def Baseline_Model_svm(type_, cv, X_train, y_train, X_test, y_test, xftitel, xtidata):
    return _baseline_model_report(SVC(probability=True), "SVC", type_, X_train, y_train, X_test, y_test,
                                  xftitel, xtidata)


def Baseline_MLPClassifier(type_, cv, X_train, y_train, X_test, y_test, xftitel, xtidata):
    return _baseline_model_report(MLPClassifier(max_iter=1000), "MLP", type_, X_train, y_train, X_test, y_test,
                                  xftitel, xtidata)

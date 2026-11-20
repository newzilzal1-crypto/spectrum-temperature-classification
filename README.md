# Classification of Optical Spectrum Measurements into Temperature Classes Using Machine Learning and Deep Learning

**Candidate:** Faraqid Qasim Mohammed  
**Supervisor:** Prof. Mohammed Abid

This repository contains the code, data, trained models and an interactive presentation app for a
thesis that classifies optical spectrum measurements into four temperature classes. Six classical
machine-learning classifiers and two deep-learning architectures are compared under a leakage-free
evaluation protocol.

## Results

Held-out test set (n = 241), sorted by macro F1-score:

| Model | Family | Accuracy | Macro F1 | Cohen's κ | MCC | ROC-AUC | 5-fold CV F1 (mean ± SD) |
|---|---|---|---|---|---|---|---|
| **MLP** | ML | **0.772** | **0.771** | **0.686** | **0.688** | **0.933** | **0.807 ± 0.018** |
| XGBoost | ML | 0.743 | 0.741 | 0.650 | 0.650 | 0.915 | 0.713 ± 0.026 |
| Decision Tree | ML | 0.726 | 0.717 | 0.627 | 0.628 | 0.813 | 0.682 ± 0.023 |
| CNN | DL | 0.656 | 0.646 | 0.533 | 0.533 | 0.863 | 0.625 ± 0.018 |
| CNN-LSTM | DL | 0.656 | 0.646 | 0.537 | 0.540 | 0.852 | 0.616 ± 0.052 |
| Random Forest | ML | 0.656 | 0.642 | 0.531 | 0.532 | 0.856 | 0.626 ± 0.034 |
| Gaussian Naive Bayes | ML | 0.577 | 0.582 | 0.447 | 0.472 | 0.765 | 0.601 ± 0.050 |
| SVM | ML | 0.618 | 0.565 | 0.468 | 0.478 | 0.848 | 0.560 ± 0.026 |

Key findings:

- **MLP** is the best model on both the test set and in cross-validation. McNemar's exact test shows it is
  significantly better (α = 0.05) than CNN, CNN-LSTM, Random Forest, Naive Bayes and SVM; the differences
  with XGBoost and Decision Tree are not significant.
- **Class4** is recognised almost perfectly by every model; **Class2** is the hardest class. Half of all
  errors are confusions between Class1 and Class2.
- Tree ensembles reach >99% training accuracy but lose 26–34 points on the test set (over-fitting).
- With 560 training samples and 12 tabular features, classical models match or beat the deep networks.

## Dataset

`Data/Data.xlsx`: 801 measurements, 12 spectral features (centroid / peak wavelength, peak level, FWHM,
OSNR, noise, …) and a `Class` label.

| Class | Temperature | Samples |
|---|---|---|
| Class1 | 0–20 °C | 286 |
| Class2 | 40 °C | 167 |
| Class3 | 60 °C | 164 |
| Class4 | 80–100 °C | 184 |

> **Target leakage:** the `temperature` column determines the class exactly, so it is **excluded** from
> the model inputs. With it, any model scores ≈100% without using the spectrum.

## Method

1. Stratified 70/30 train/test split (seed 42).
2. IQR outlier clipping (1.5 × IQR) and standardisation, **fitted on the training data only**
   (re-fitted inside every cross-validation fold).
3. 5-fold stratified cross-validation on the training set, then final training on the whole training set.
4. Evaluation on the held-out test set: accuracy, macro precision / recall / F1, Cohen's κ, MCC,
   one-vs-rest ROC-AUC, confusion matrices and McNemar's exact test.

Models: XGBoost, Random Forest, Gaussian Naive Bayes, Decision Tree, SVM (RBF), MLP, a 1-D CNN and a
CNN-LSTM (Adam, 100 epochs, batch size 64). All classical models use library-default hyper-parameters.

## Getting started

```bash
git clone <repository-url>
cd <repository-folder>
pip install -r requirements.txt
streamlit run app.py
```

The trained models are included in `saved_models/`, so the app starts immediately. To retrain everything
(about 5 minutes on a CPU):

```bash
python train_models.py            # full run: 5-fold CV + final models
python train_models.py --epochs 5 --folds 3   # quick check
```

### The app

| Tab | Content |
|---|---|
| 1 · Overview | Abstract, key results, objectives, class definitions |
| 2 · Methodology | Dataset statistics, pipeline diagram, models, metric definitions |
| 3 · Results | Test and CV tables, ROC curves, confusion matrices, per-class F1, McNemar tests |
| 4 · Live testing | Run any model on the test set, a single measurement or an uploaded file |
| 5 · Discussion & Conclusion | Discussion, limitations, conclusion, future work, report export |

The **Academic report** button in tab 5 exports a self-contained HTML report (open it in a browser and
use *Print → Save as PDF*). A pre-generated copy is in `thesis_results_report.html`. Title, names and
institution shown in the app and report are read from `project_info.json`.

## Repository structure

```
app.py                 Streamlit presentation app
pipeline.py            Shared preprocessing, model definitions and metrics
train_models.py        Cross-validation + training; writes saved_models/
report.py              Printable academic HTML report
project_info.json      Thesis title, candidate, supervisor, institution
Data/Data.xlsx         Dataset
saved_models/          Trained models, preprocessor, data splits, CV results
.streamlit/config.toml App theme

Classfaction ML.py     Original research scripts (classical ML, CNN, CNN-LSTM),
Classfaction DL CNN.py write their figures and reports to System_*/ folders
Classfaction DL RNN.py
init_paramaters.py, utilities_ML.py, utilities_DL.py, Create_mode_DL.py
```

## Notes

- On machines with many CPU cores and little free memory, OpenBLAS can fail to allocate its thread
  buffers. The entry scripts therefore cap `OPENBLAS_NUM_THREADS` / `OMP_NUM_THREADS` at 4 unless you
  set them yourself.
- Results are reproducible with the fixed seed (42); deep-learning results can vary slightly between
  machines because of floating-point non-determinism.

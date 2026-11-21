# IoT and Artificial Intelligence to Design and Implement a Monitoring System for Distribution Transformers Based on an Optical Fiber Fabry–Perot Sensor with Micro-Cavity

*AI module: classification of transformer-oil condition from optical spectrum measurements*

**Candidate:** Faraqid Qasim Mohammed  
**Supervisors:** Prof. Mohamed Abid & Dr. Yassine Aydi  
**Institution:** National School of Electronics and Telecommunications of Sfax (ENET'Com), University of Sfax — Ph.D. thesis, 2026

This repository contains the code, data, trained models and an interactive presentation app for the
AI module of the thesis: classifying transformer-oil condition (**Good, Not Bad, Poor, Very Poor**) from
measurements of an optical-fibre Fabry–Perot sensor with air micro-cavities (FBG–PCF–FBG). Six classical
machine-learning classifiers and two deep-learning models are compared under two protocols.

## Results

Held-out test set (n = 241), sorted by macro F1-score.

### Thesis protocol: all 13 attributes (as in Chapter 4)

| Model | Family | Accuracy | Macro F1 | Cohen's κ | ROC-AUC | Errors | 5-fold CV F1 |
|---|---|---|---|---|---|---|---|
| **XGBoost** | ML | **1.000** | **1.000** | **1.000** | **1.000** | 0 | 1.000 ± 0.000 |
| **Decision Tree** | ML | **1.000** | **1.000** | **1.000** | **1.000** | 0 | 1.000 ± 0.000 |
| Gaussian Naive Bayes | ML | 0.992 | 0.992 | 0.989 | 0.993 | 2 | 0.995 ± 0.005 |
| SVM | ML | 0.992 | 0.992 | 0.989 | 1.000 | 2 | 0.998 ± 0.004 |
| Random Forest | ML | 0.992 | 0.991 | 0.989 | 0.999 | 2 | 1.000 ± 0.000 |
| MLP | ML | 0.992 | 0.991 | 0.989 | 1.000 | 2 | 1.000 ± 0.000 |
| CNN-LSTM | DL | 0.975 | 0.977 | 0.966 | 1.000 | 6 | 0.957 ± 0.050 |
| CNN | DL | 0.975 | 0.974 | 0.966 | 0.999 | 6 | 0.988 ± 0.018 |

This confirms the near-100% accuracy reported in the thesis.

### Spectrum-only protocol: temperature excluded (robustness analysis)

The classes are defined by thresholds on the oil temperature, so with temperature as an input a model can
reach ≈100% by learning those thresholds. Repeating the experiment **without temperature** shows that the
optical spectrum alone carries oil-condition information:

| Model | Family | Accuracy | Macro F1 | Cohen's κ | ROC-AUC | 5-fold CV F1 |
|---|---|---|---|---|---|---|
| **MLP** | ML | **0.772** | **0.771** | **0.686** | **0.933** | **0.807 ± 0.018** |
| XGBoost | ML | 0.743 | 0.741 | 0.650 | 0.915 | 0.713 ± 0.026 |
| Decision Tree | ML | 0.726 | 0.717 | 0.627 | 0.813 | 0.682 ± 0.023 |
| CNN-LSTM | DL | 0.664 | 0.642 | 0.538 | 0.857 | 0.649 ± 0.030 |
| Random Forest | ML | 0.656 | 0.642 | 0.531 | 0.856 | 0.626 ± 0.034 |
| CNN | DL | 0.647 | 0.640 | 0.523 | 0.852 | 0.651 ± 0.050 |
| Gaussian Naive Bayes | ML | 0.577 | 0.582 | 0.447 | 0.765 | 0.601 ± 0.050 |
| SVM | ML | 0.618 | 0.565 | 0.468 | 0.848 | 0.560 ± 0.026 |

Chance level is 25% and the majority-class baseline is 35.7%. Half of the remaining errors are confusions
between Good and Not Bad.

## Dataset

`Data/Data.xlsx`: 801 records (805 collected, 4 with missing values removed), the oil temperature, 12
optical attributes of the reflected spectrum (centroid / peak wavelength, peak level, FWHM, OSNR, noise, …)
and a `Class` label. Data were collected with the Diyala State Company for Electrical Industries and
measured at the Laser Institute for Postgraduate Studies, University of Baghdad.

| Class (code) | Oil condition | Labelling rule | Measured temperatures | Samples |
|---|---|---|---|---|
| Class1 | Good | 0 < T ≤ 25 °C | 0–20 °C | 286 |
| Class2 | Not Bad | 25 < T ≤ 50 °C | 40 °C | 167 |
| Class3 | Poor | 50 < T ≤ 75 °C | 60 °C | 164 |
| Class4 | Very Poor | 75 < T ≤ 100 °C | 80–100 °C | 184 |

## Method

1. Stratified 70/30 train/test split (seed 42); the networks use 20% of the training data for validation.
2. IQR outlier clipping (1.5 × IQR) and z-score normalisation, **fitted on the training data only**
   (re-fitted inside every cross-validation fold).
3. 5-fold stratified cross-validation on the training set, then final training on the whole training set.
4. Evaluation on the held-out test set: accuracy, macro precision / recall / F1, Cohen's κ, MCC,
   one-vs-rest ROC-AUC, confusion matrices and McNemar's exact test.
5. Everything is run twice: with all 13 attributes and without temperature.

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
(about 10 minutes on a CPU, both protocols):

```bash
python train_models.py            # full run, both protocols: 5-fold CV + final models
python train_models.py --epochs 5 --folds 3   # quick check
```

### The app

| Tab | Content |
|---|---|
| 1 · Overview | Abstract, key results, objectives, class definitions |
| 2 · Methodology | Dataset statistics, pipeline diagram, models, metric definitions |
| 3 · Results | Test and CV tables, ROC, confusion matrices, McNemar tests; 3.8 spectrum-only analysis |
| 4 · Live testing | Run any model (either feature set) on the test set, a single measurement or a file |
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
saved_models/thesis/   Trained models, preprocessor, splits, CV results (all 13 attributes)
saved_models/spectrum/ The same without temperature
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

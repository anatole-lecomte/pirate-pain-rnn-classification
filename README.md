# Multivariate Time Series Classification — RNN Architectures Compared

Six recurrent neural network architectures (RNN, BiRNN, GRU, BiGRU, LSTM, BiLSTM), trained entirely from scratch, benchmarked on a multivariate time series classification task from a Kaggle competition.

*University project — Artificial Neural Networks and Deep Learning (AN2DL), Politecnico di Milano (2025–2026). Team "The Underfitters", 3 students.*

## Overview

The task ([Kaggle competition](https://www.kaggle.com/competitions/an2dl2526c2/overview)) is to classify the pain level of subjects ("Pirates") — `no_pain`, `low_pain`, or `high_pain` — from a multivariate physiological time series of 160 time steps per sample, mixing continuous features (joint angles, pain survey scores) with categorical/textual attributes (e.g. number of eyes or hands given as words). No pretrained architectures or weights were allowed; every model was trained from scratch, and the competition metric was F1-score, which put a premium on handling the class imbalance (~77% no_pain, 14% low_pain, 8% high_pain) correctly.

## Key results

| Model | LSTM | Bi-LSTM | RNN | Bi-RNN | GRU | Bi-GRU |
|---|---|---|---|---|---|---|
| F1-score | 0.9358 | 0.9200 | 0.7085 | 0.8942 | **0.9304** | **0.9356** |
| Accuracy | 0.9357 | 0.9241 | 0.7731 | 0.8975 | 0.9337 | 0.9378 |

- **Best model on the held-out test set: GRU** (window size 100, stride 20, learning rate 1e-3, 5 layers × 128 units), reaching an **F1-score of 92.95%**.
- **LSTM and Bi-GRU were the strongest architectures overall** on the validation split; plain **RNN consistently underperformed**, with flat or unstable training curves suggesting it was insufficiently expressive for this task.
- Bidirectional variants generally outperformed their unidirectional counterparts, except for LSTM where the plain version edged out Bi-LSTM.

## Method highlights

- **Preprocessing**: categorical/textual attributes (e.g. word-based counts of eyes/hands/legs) mapped to numerical values, min-max normalization of all features, stratified train/validation split preserving class proportions.
- **Windowing**: configurable window size and stride to convert each 160-step recording into fixed-length model inputs.
- **Model experimentation**: six recurrent architectures implemented and tuned manually across layer count, hidden size, learning rate, batch size, dropout, L1/L2 regularization, and early-stopping patience. A grid search was also attempted but was dropped — it did not outperform manual tuning while being far more expensive on Colab's free GPU quota.
- **K-fold cross-validation**: used to reduce the bias of a single train/validation split and get a more robust estimate of model performance.
- **TensorBoard**: training curves for all six architectures logged and compared side by side.

## Honest limitations (from the report's discussion)

- A systematic (rather than manual) hyperparameter grid search would likely have improved results further, but was computationally prohibitive within the available GPU quota.
- Class proportions between train and validation splits were close but not perfectly balanced.
- The k-fold cross-validation results were not fully exploited: the best hyperparameters were not used to retrain a final model on the *full* training set before submission, which likely left some test-set performance on the table.

## Tech stack

`TensorFlow / Keras` · `NumPy` / `pandas` · `scikit-learn` (stratified splitting, k-fold) · `TensorBoard`

## Repository structure

```
.
├── README.md
├── ANN_DL_assignement1.ipynb   # Full notebook: preprocessing, all 6 architectures, cross-validation, submission
└── The_Underfitters.pdf         # Full write-up: problem analysis, method, results table, discussion
```

The notebook was built for Google Colab (Kaggle API download, GPU runtime) and expects a Kaggle API token to fetch the competition dataset, which is not included in this repository.

**Note on licensing**: parts of the course-provided starter code in this notebook are licensed under Apache License 2.0 (© 2025 Eugenio Lomurno, Alberto Archetti, Roberto Basla, Carlo Sgaravatti) — this attribution is kept intact in the notebook as required by that license.

## Authors
Politecnico di Milano — Artificial Neural Networks and Deep Learning
- BENKIRANE Ilyas
- LECOMTE Anatole
- LUNEAU Nathan

  


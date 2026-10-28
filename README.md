# Resume Domain Classification Project

This repository contains a resume-domain classification workflow for multi-class job category prediction from raw CV text. The project includes a deliberately overfit Vanilla LSTM baseline, a stronger pooled LSTM comparison model, and experiment artifacts for benchmarking model quality and generalization.

The project is designed to demonstrate both:

- the failure mode of an unregularized recurrent model on a noisy text classification task
- the substantial improvement gained by using a pooled sequence representation with a matched architecture

---

## Overview

The goal of this project is to classify resumes into one of 24 professional domains using cleaned text from CV documents. This is a text classification problem with class imbalance, long sequences, and domain-specific vocabulary.

The repository includes:

- a preprocessing and stratified splitting pipeline
- repeated experiments with LSTM-based models
- evaluation metrics for train/validation/test performance
- confusion-matrix and training-curve visualizations
- a comparison between a standard LSTM and a pooled LSTM

---

## Repository Structure

```text
nlp-cv/
├── README.md
├── requirements.txt
├── data/
│   └── .gitkeep
├── models/
│   ├── baseline_metrics.json
│   ├── confusion_matrix.png
│   ├── history.json
│   ├── history_pooled_lstm.json
│   ├── normal_lstm_matched.pt
│   ├── normal_lstm_matched_history.json
│   ├── normal_vs_pooled_lstm_comparison.csv
│   ├── normal_vs_pooled_lstm_test_metrics.png
│   ├── normal_vs_pooled_lstm_training_curves.png
│   ├── overfitting_curves.png
│   ├── pooled_lstm_confusion_matrix.png
│   ├── pooled_lstm_metrics.json
│   ├── pooled_lstm_test_metrics.png
│   ├── pooled_lstm_training_curves.png
│   ├── vanilla_lstm_overfit.pt
│   └── vanilla_lstm_pooled_lstm.pt
├── notebooks/
│   ├── README.md
│   └── resume_lstm_pipeline_comparison_focused.ipynb
└── .gitignore
```

---

## Pipeline

### 1) Data preparation
The workflow begins with text cleaning and normalization:

- lowercasing
- handling Unicode/full-width character issues
- removing formatting artifacts and bullets
- whitespace normalization
- preservation of technical tokens such as C++, C#, and R&D

The processed text is then split into train/validation/test sets using a stratified approach to preserve label balance.

### 2) Vocabulary and label encoding
The vocabulary is built from the training split only to prevent leakage. The label encoder maps the 24 target classes to integer IDs.

### 3) Sequence representation
Each resume is tokenized and padded/truncated to a fixed length of 400 tokens, producing integer sequences ready for PyTorch training.

### 4) Model training
The project compares:

- a standard Vanilla LSTM which uses the final hidden state for classification
- a pooled LSTM which combines mean and max pooling across the time dimension

---

## Dataset

This project is built for resume-domain classification using real-world CV text. The intended data layout is:

```text
data/
├── raw/
│   └── resume.csv
├── preprocessed/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   ├── vocab.json
│   └── label_encoder.json
```

The original dataset contains resumes across 24 categories, with the final preprocessing pipeline producing train/validation/test splits suitable for benchmarking machine learning models.

---

## Experimental Highlights

### Vanilla LSTM overfitting baseline
The baseline model is intentionally trained with no regularization to highlight memorization behavior and overfitting.

Key observations:

- train accuracy rises to around 91.6%
- validation accuracy collapses to about 8.3%
- validation loss increases substantially after the first few epochs
- the model shows a large generalization gap and poor out-of-sample performance

This is a classic overfitting demonstration and provides the benchmark against which better models are compared.

### Pooling-based LSTM comparison
The improved architecture uses pooled LSTM representations rather than only the final hidden state. This produces much better generalization and stronger performance on the held-out test set.

---

## Results Summary

| Model | Best Validation Loss | Best Val Accuracy | Test Accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| Vanilla LSTM (final hidden state) | 3.1269 | 8.29% | 11.17% | 9.82% |
| Pooled LSTM (mean + max pooling) | 0.9175 | 75.69% | 73.86% | 66.76% |

The pooled LSTM substantially outperforms the baseline, reducing the generalization gap and delivering strong classification performance across the 24 classes.

---

## Important Metrics

From the repository artifacts:

- Vanilla LSTM parameters: 2,847,768
- Pooled LSTM parameters: 2,058,264
- Pooled LSTM test accuracy: 73.86%
- Pooled LSTM macro F1: 66.76%
- Generalization gap for pooled LSTM: ~17.66%

These numbers show that the improved architecture successfully moves the system from severe overfitting toward a practical classification model.

---

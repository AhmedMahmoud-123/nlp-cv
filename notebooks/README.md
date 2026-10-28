
---


##  Section-by-Section Summary

| Section | Title | What it does | Key Output |
|---|---|---|---|
| **0** | **Config & Imports** | Imports PyTorch, pandas, numpy, sklearn, matplotlib; sets `SEED=42`, paths, and zero-regularization parameters | Reproducible environment |
| **1** | **Preprocessing & Splitting** | Cleans raw text, truncates to 400 tokens, performs 70/15/15 stratified split, builds 15,000 vocab from train set | `data/preprocessed/` CSVs, `vocab.json`, `label_encoder.json` |
| **2** | **Dataset & DataLoaders** | Implements `ResumeDataset` and PyTorch `DataLoader` factory with `batch_size=64` | `train_loader` (27 batches), `val_loader` (6), `test_loader` (7) |
| **3** | **Model Architecture** | Constructs 2-layer Vanilla LSTM (hidden size 256, 128d embedding) with 2,847,768 trainable parameters | `VanillaLSTMClassifier` |
| **4** | **Overfitting Experiment** | Trains for 12 epochs with 0.0 dropout and 0.0 weight decay to demonstrate loss divergence and memorization | `models/vanilla_lstm_overfit.pt`, `models/history.json` |
| **5** | **Evaluation & Artifacts** | Evaluates on 394 unseen test resumes, computes Macro F1 and per-class reports, and plots loss/accuracy curves | `models/overfitting_curves.png`, `models/confusion_matrix.png`, `models/baseline_metrics.json` |

---

## Overfitting Experiment Results

| Metric | Result | Analysis |
|---|---|---|
| **Train Accuracy** | **91.67%** | Model memorized training resumes |
| **Validation Accuracy** | **8.29%** | Severe failure to generalize |
| **Generalization Gap** | **83.38%** | **Textbook overfitting demonstration** |
| **Best Val Loss** | **3.1269** (Epoch 2) | Optimal stopping point before divergence |
| **Final Val Loss** | **5.2308** | Diverged by **+2.1039** |
| **Test Accuracy** | **10.15%** | Baseline for Step 3 paper optimizations |
| **Test Macro F1** | **9.01%** | 7 minority/overlapping classes collapsed to 0.0% |



# Egyptian Currency CNN Classifier

CNN-based classifier for Egyptian polymer banknotes (10 EGP and 20 EGP) using TensorFlow/Keras. Built from scratch (no transfer learning). Target: 93% validation accuracy.

## Architecture

4 Conv2D blocks with BatchNormalization, dropout, and a Dense(512) head.

## Dataset

[Egyptian New Currency 2023](https://www.kaggle.com/datasets/belalsafy/egyptian-new-currency-2023) from Kaggle.

## Runs

| Run | Config | Best Val Acc | Status |
|-----|--------|-------------|--------|
| 1 | Baseline (no BatchNorm) | 53.6% | COMPLETE |
| 2 | +BatchNorm | 65.4% | COMPLETE |
| 3 | L2(0.01) + LR(5e-5) | 15.0% | CATASTROPHE |
| 4 | Revert to Run 2 + more aug | 69.0% | PARTIAL (8/10) |

See RUN_TRACKING.md for full details.

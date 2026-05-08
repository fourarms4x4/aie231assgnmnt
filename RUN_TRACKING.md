# Egyptian Currency CNN - COMPLETE RUN TRACKING

================================================================================
PROJECT: Egyptian Currency CNN Classifier (2023 Polymer Banknotes)
TARGET: 93% validation accuracy on 10 EGP and 20 EGP polymer notes
================================================================================

================================================================================
QUICK STATUS
================================================================================

| Run | Config | Best Val Acc | Date | Status |
|-----|--------|-------------|------|--------|
| 1 | Baseline (no BatchNorm) | 53.6% | Apr 25 | COMPLETE |
| 2 | +BatchNorm (your arch) | 65.4% | Apr 25 | COMPLETE |
| 3 | +L2(0.01) + LR(5e-5) | 15.0% | Apr 25 | CATASTROPHE |
| 4 | REVERT to Run 2 + more aug | 69.0% | Apr 25 | PARTIAL (8/10) |

**CURRENT BEST: 69.01% (Run 4, Run 5)**

Gap to target: 24%

================================================================================
FULL RUN HISTORY
================================================================================

## CLASSES IN DATASET (9 total)
['1', '10', '10 (new)', '100', '20', '20 (new)', '200', '5', '50']

Target polymer notes: '10 (new)', '20 (new)'

---

## RUN 1: BASELINE (COMPLETE)
Date: 2026-04-25

**FULL CONFIG:**
- Architecture: 4 Conv2D blocks
  - Conv2D(32, 3×3) → MaxPool
  - Conv2D(64, 3×3) → MaxPool
  - Conv2D(128, 3×3) → MaxPool → Dropout(0.2)
  - Conv2D(256, 3×3) → MaxPool → Dropout(0.2)
  - Flatten
  - Dense(512) → Dropout(0.5)
  - Dense(9, softmax)
- Augmentation:
  - RandomFlip("horizontal")
  - RandomRotation(0.1)
  - RandomZoom(0.1)
- Optimizer: Adam (default lr=0.001)
- Rescaling: 1./255
- Epochs: 15
- Batch size: 32
- Image size: 224×224
- Validation split: 20%

**FULL RESULTS:**
- Training Accuracy: 53.2%
- Validation Accuracy: 50.4%
- Best Validation: 53.6% (epoch 14)

**PER-CLASS (from confusion matrix):**
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|---------|
| 1 | 1.00 | 0.43 | 0.60 |
| 10 | 0.33 | 0.34 | 0.34 |
| 10 (new) | 0.87 | 0.49 | 0.63 |
| 100 | 0.43 | 0.35 | 0.39 |
| 20 | 0.34 | 0.77 | 0.47 |
| 20 (new) | 0.79 | 0.80 | 0.80 |
| 200 | 0.63 | 0.30 | 0.40 |
| 5 | 0.46 | 0.45 | 0.46 |
| 50 | 0.48 | 0.46 | 0.47 |

Issues:
- No BatchNormalization
- Default LR too high
- Underfitting

---

## RUN 2: +BATCHNORM (YOUR SPECIFIED ARCHITECTURE - COMPLETE)
Date: 2026-04-25

**FULL CONFIG (YOUR EXACT SPEC):**
- Architecture: 4 Conv2D blocks + BatchNorm
  - Conv2D(32, 3×3, padding='same') → BatchNorm → MaxPool
  - Conv2D(64, 3×3, padding='same') → BatchNorm → MaxPool
  - Conv2D(128, 3×3, padding='same') → BatchNorm → MaxPool → Dropout(0.2)
  - Conv2D(256, 3×3, padding='same') → BatchNorm → MaxPool → Dropout(0.3)
  - Flatten
  - Dense(512) → BatchNorm → Dropout(0.5)
  - Dense(9, softmax)
- Augmentation:
  - RandomFlip("horizontal_and_vertical")
  - RandomRotation(0.2)
  - RandomZoom(0.2)
- Optimizer: Adam (lr=1e-4)
- Rescaling: 1./255
- Epochs per run: 25
- Batch size: 32
- Image size: 224×224
- Callbacks:
  - EarlyStopping (monitor="val_accuracy", patience=8, restore_best_weights=True)
  - ReduceLROnPlateau (monitor="val_loss", factor=0.5, patience=4)

**ALL 10 RUNS COMPLETE RESULTS:**
┌─────┬───────────┬───────┬───────────┬──────────┐
│ Run │  Best Val │ Epoch │ Final Val │ Train Acc│
├─────┼───────────┼───────┼───────────┼──────────┤
│  1  │  57.03%   │  25   │  57.03%   │  ~72%    │
│  2  │  61.79%   │  20   │  60.08%   │  ~75%    │
│  3  │  65.21%   │  25   │  65.21%   │  ~78%    │
│  4  │  65.40%   │  24   │  61.03%   │  ~77%    │ ← BEST
│  5  │  58.75%   │  24   │  57.22%   │  ~71%    │
│  6  │  55.70%   │  22   │  55.13%   │  ~70%    │
│  7  │  60.65%   │  16   │  54.75%   │  ~73%    │
│  8  │  60.08%   │  13   │  59.89%   │  ~74%    │
│  9  │  61.79%   │  16   │  60.46%   │  ~75%    │
│ 10  │  57.98%   │  24   │  57.03%   │  ~72%    │
└─────┴───────────┴───────┴───────────┴──────────┘

Statistics:
- Mean Best: 59.94%
- Std Dev: 3.3%

Best: 65.40% (Run 4)
Gap to target: 27.6%

---
## RUN 3: +L2 REGULARIZATION (CATASTROPHE)
Date: 2026-04-25

**FULL CONFIG:**
- Architecture: 4 Conv2D blocks + L2(0.01) + BatchNorm
  - Conv2D(32, 3×3, kernel_regularizer=l2(0.01)) → BatchNorm → MaxPool → Dropout(0.3)
  - Conv2D(64, 3×3, kernel_regularizer=l2(0.01)) → BatchNorm → MaxPool → Dropout(0.4)
  - Conv2D(96, 3×3, kernel_regularizer=l2(0.01)) → BatchNorm → MaxPool → Dropout(0.5)
  - Conv2D(128, 3×3, kernel_regularizer=l2(0.01)) → BatchNorm → MaxPool → Dropout(0.6)
  - Dense(256, kernel_regularizer=l2(0.01)) → Dropout(0.6)
  - Dense(128, kernel_regularizer=l2(0.01)) → Dropout(0.5)
  - Dense(9, softmax)
- Augmentation:
  - RandomFlip("horizontal_and_vertical")
  - RandomRotation(0.3)
  - RandomZoom(0.2)
  - RandomBrightness(0.15)
  - RandomContrast(0.15)
  - RandomTranslation(0.1)
- Optimizer: Adam (lr=5e-5) ← TOO LOW!
- Epochs: 30 per run

**ALL 10 RUNS - JUST ABOVE RANDOM:**
┌─────┬───────────┬───────┬───────────┐
│ Run │  Best Val │ Epoch │ Final Val │
├─────┼───────────┼───────┼───────────┤
│  1  │  12.74%   │  18   │  11.60%   │
│  2  │  14.64%   │  23   │  13.12%   │
│  3  │  14.45%   │   1   │  13.69%   │
│  4  │  14.45%   │   3   │  13.12%   │
│  5  │  12.93%   │   7   │  12.36%   │
│  6  │  15.02%   │  18   │  15.02%   │ ← BEST (lol)
│  7  │  15.02%   │  13   │  13.50%   │
│  8  │  13.31%   │  16   │  11.60%   │
│  9  │  13.50%   │  20   │  12.74%   │
│ 10  │  13.69%   │   9   │  13.12%   │
└─────┴───────────┴───────┴───────────┘

Statistics:
- Mean Best: 13.9%
- Random guess: 11.1% (9 classes)
- Only 2.8% above random!

BEST: 15.02% (Run 6)
Gap to target: 78% ← CATASTROPHE

**WHY IT FAILED:**
1. L2(0.01) crushed ALL weights every step
2. LR(5e-5) too low to overcome L2 penalty
3. Combined effect: model stopped learning
4. Lesson: NEVER use strong L2 + low LR together!

---
## RUN 4: REVERT TO RUN 2 + MORE AUG (PARTIAL - 8/10 RUNS)
Date: 2026-04-25
Session died at Run 9 (quota exhausted)

**FULL CONFIG:**
- Architecture: REVERTED to Run 2 (NO L2!)
  - Conv2D(32, 3×3, padding='same') → BatchNorm → MaxPool
  - Conv2D(64, 3×3, padding='same') → BatchNorm → MaxPool
  - Conv2D(128, 3×3, padding='same') → BatchNorm → MaxPool → Dropout(0.2)
  - Conv2D(256, 3×3, padding='same') → BatchNorm → MaxPool → Dropout(0.3)
  - Flatten
  - Dense(512) → BatchNorm → Dropout(0.5)
  - Dense(9, softmax)
- Augmentation (ADDED):
  - RandomFlip("horizontal_and_vertical")
  - RandomRotation(0.2)
  - RandomZoom(0.2)
  - RandomBrightness(0.1) ← NEW
  - RandomContrast(0.1) ← NEW
- Optimizer: Adam (lr=1e-4) ← SAME AS RUN 2, NOT 5e-5!
- Epochs: 30 per run

**RESULTS (Runs 1-8 of 10):**
┌─────┬───────────┬───────┬───────────┬──────────┬────────────┐
│ Run │  Best Val │ Epoch │ Final Val │ Train Acc │  Status   │
├─────┼───────────┼───────┼───────────┼──────────┼────────────┤
│  1  │  49.24%   │  17   │  49.05%   │  69.12%   │ Complete │
│  2  │  53.80%   │  28   │  53.23%   │  74.16%   │ Complete │
│  3  │  61.03%   │  30   │  61.03%   │  80.14%   │ Complete │
│  4  │  67.11%   │  30   │  67.11%   │  74.25%   │ Complete │
│  5  │  69.01%   │  28   │  68.06%   │  79.43%   │ BEST ★   │
│  6  │  52.28%   │  14   │  51.33%   │  67.84%   │ Complete │
│  7  │  56.27%   │  26   │  53.99%   │  70.40%   │ Complete │
│  8  │  61.98%   │  29   │  60.08%   │  74.77%   │ Complete │
└─────┴───────────┴───────┴───────────┴──────────┴────────────┘

Session DIED (quota) at Run 9!

Statistics (8 runs):
- Mean Best: 59.0%
- Best: 69.01% (Run 5) ★ NEW BEST!
- Worst: 49.24% (Run 1)

Gap to target: 23.99%

**BEST RESULT: 69.01% (Run 5)** ← beats Run 2!

**WHY IT WORKED:**
- Reverted L2 (disaster in Run 3)
- Kept LR at 1e-4 (not 5e-5!)
- Added brightness/contrast augmentation
- Same BatchNorm + dropout from Run 2

================================================================================
WHAT WAS LOST
================================================================================

When Colab quota depletes, session is wiped:
- Run 4, Runs 9-10 not completed
- Best model from Run 5 NOT downloaded

File that SHOULD exist on device:
- visualizations/run4_[timestamp]/ (if saved manually)

Files NOT saved (need to recreate):
- Best model (.keras)
- Run comparison plots
- Confusion matrix
- Accuracy/loss curves

================================================================================
OPTIONS FOR LATER
================================================================================

OPTION A: COMPLETE RUN 4 (RECOMMENDED)
----------------------------------
- Just run Runs 9-10 to finish batch
- Expected: ~70% or higher
- Quick fix, minimal time

To do:
1. Load notebook
2. Run cells 1-10
3. Model will load Run 4 config (already in notebook!)

---

OPTION B: USE EXISTING MODEL FROM RUN 5
----------------------------------
- Already have 69% model saved (if user downloaded)
- Try optimizations:
  - Label smoothing
  - MixUp augmentation
  - Test-time augmentation (TTA)

---

OPTION C: START FRESH
-------------------
- Run COMPLETE Run 4 again (10 runs)
- Could potentially exceed 70%

================================================================================
WHAT WE LEARNED
================================================================================

1. BatchNorm ESSENTIAL (+11.8% from Run 1→2)

2. L2 REGULARIZATION DANGEROUS
   - L2(0.01) = CATASTROPHE (15%)
   - Never use strong L2 + low LR together

3. LEARNING RATE MATTERS
   - 1e-4 works (Run 2, 4)
   - 5e-5 kills learning (Run 3)

4. AUGMENTATION HELPS
   - Adding brightness/contrast improved 65%→69%

5. BEST CONFIG SO FAR:
   - Your specified architecture + BatchNorm
   - LR: 1e-4
   - Augmentation: Flip, Rotation, Zoom, Brightness, Contrast
   - NO L2!

================================================================================
NEXT STEPS (WHEN YOU HAVE GPU ACCESS AGAIN)
================================================================================

1. QUICK: Complete Run 4 (runs 9-10)
   - Should get to ~70%+

2. IF 70% STILL NOT ENOUGH:
   - Try label smoothing (0.1)
   - Try MixUp augmentation
   - Try TTA (test-time augmentation)

3. KEY CONSTRAINT:
   - NO TRANSFER LEARNING allowed
   - Must build from scratch

4. END GOAL:
   - 93% on validation
   - Focus: polymer 10 EGP and 20 EGP

================================================================================
NOTES FOR PDF REPORT (DELIVERABLE)
================================================================================

Required for 1-page PDF:
- [ ] Architecture diagram
- [ ] Loss/accuracy curves (best run)
- [ ] Confusion matrix
- [ ] Example predictions
- [ ] This run history table

Run 4 produces best results so far at 69%

===============================================================================
RUN 5 PLAN: TARGET 93%
===============================================================================

**Date**: 2026-05-08
**Target**: 93% validation accuracy
**Strategy**:
- Architecture: 2 conv layers per block, 32→64→128→256 filters
- Head: Dense(512) → Dense(256) for extra capacity
- Optimizer: SGD (momentum=0.9) with CosineDecay LR (0.01 → 1e-5)
- Label smoothing: 0.1
- Dropout: increased (0.25, 0.35, 0.5, 0.5 conv; 0.5, 0.3 dense)
- Epochs: 100 per run × 10 runs
- Ensemble: top 3 models averaged
- Status: Notebook ready, awaiting Colab GPU

===============================================================================
END OF TRACKING
===============================================================================

Last Updated: 2026-05-08
Current Status: Run 5 notebook ready for Colab T4 GPU
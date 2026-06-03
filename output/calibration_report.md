# Calibration Fix Report: Orphan GPCR Deorphanization

## Critical Finding

The model's confidence scores are **severely overconfident**:

- At confidence >0.8, actual accuracy = 0.0
- At confidence >0.9, actual accuracy = 0.0
- ECE (all predictions) = 0.1382 (moderate)
- ECE (top-5 predictions) = 0.1784 (poor)
- Brier score = 0.1084

This means the model is **most confident when it is most wrong**.

## Calibration Methods Compared

| Method | ECE | ECE (top-5) | Brier Score |
|--------|-----|-------------|-------------|
| uncalibrated | 0.1382 | 0.1784 | 0.1084 |
| isotonic_regression | 0.8423 | 0.8880 | 0.8423 |
| platt_scaling | 0.0032 | 0.0013 | 0.0206 |
| temperature_scaling | 0.5210 | 0.5543 | 0.3159 |

## Recommended Approach

- **Best ECE:** platt_scaling (ECE = 0.0032)
- **Best Brier:** platt_scaling (Brier = 0.0206)

## Key Insight

Calibration **cannot fix** the fundamental limitation of this model: the underlying predictions are weak (P@5 = 10.68%, AUC-ROC = 0.5878). Calibration only makes the confidence scores honest; it does not improve the underlying prediction quality. The model's predictions should be treated as hypotheses for experimental testing, not as reliable probability estimates.

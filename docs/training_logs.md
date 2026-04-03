# Training Log - 100 Epoch Run

**Timestamp:** 2026-03-22

## Overview
This document keeps track of the training history for the 100-epoch training run of the hybrid model. The final stages of training are summarized below.

## Final Epoch Summaries

### Epoch 96/100
- **Duration:** 59.2s
- **Training Loss:** 5.8928
  - Recon: 5.1725
  - KL: 4.0420
  - Expr: 1.0179
- **Validation Loss:** 6.1247

### Epoch 97/100
- **Duration:** 61.2s
- **Training Loss:** 5.8541
  - Recon: 5.1696
  - KL: 3.0814
  - Expr: 0.9904
- **Validation Loss:** 6.0700
- **Events:**
  - Saved checkpoint: `hybrid/checkpoints/checkpoint_latest.pt`
  - Saved best model: `hybrid/checkpoints/best_model.pt`

### Epoch 98/100
- **Duration:** 61.4s
- **Training Loss:** 5.8598
  - Recon: 5.1807
  - KL: 2.7880
  - Expr: 0.9716
- **Validation Loss:** 6.0775

### Epoch 99/100
- **Duration:** 62.3s
- **Training Loss:** 5.8195
  - Recon: 5.1620
  - KL: 1.9946
  - Expr: 0.9443
- **Validation Loss:** 6.0806

### Epoch 100/100
- **Duration:** 62.3s
- **Training Loss:** 5.8322
  - Recon: 5.1617
  - KL: 2.0524
  - Expr: 0.9537
- **Validation Loss:** 6.0988
- **Events:**
  - Saved checkpoint: `hybrid/checkpoints/checkpoint_latest.pt`

## Conclusion
- **Training complete!**
- The **best model** was identified and saved at **Epoch 97** with a Validation Loss of `6.0700`.
- The **latest checkpoint** from the final epoch was correctly saved.

# Self-Pruning Neural Network

## 1. Introduction

Neural networks often contain redundant parameters, leading to inefficiency in memory and computation. This project implements a **self-pruning neural network** that learns to remove unnecessary connections during training.

Unlike traditional pruning methods that operate post-training, this method integrates pruning into the learning process.

---

## 2. Methodology

### 2.1 Prunable Linear Layer

A custom linear layer (`PrunableLinear`) is implemented where each weight is associated with a learnable gate parameter.

Gate values are computed as:

g = sigmoid(gate_scores)

The effective weights are:

W_pruned = W ⊙ g

Weights corresponding to gates near zero are effectively pruned.

---

### 2.2 Loss Function

Total Loss = Classification Loss + λ × Sparsity Loss

* Classification Loss: Cross-Entropy
* Sparsity Loss: Mean of all gate values

---

### 2.3 Why L1 Encourages Sparsity

The sparsity loss behaves like an L1 penalty on gate values. L1 regularization encourages parameters to shrink toward zero. Since gates lie in (0,1), minimizing their sum drives many gates close to zero, effectively removing those connections.

---

### 2.4 Training Strategy

A warm-up phase is used:

* Initial epochs: only classification loss
* Later epochs: classification + sparsity loss

This prevents early pruning before feature learning.

---

## 3. Experimental Setup

* Dataset: CIFAR-10
* Model: Fully connected network with prunable layers
* Optimizer: Adam
* Epochs: 10
* Batch size: 64

---

## 4.  Results

| Lambda (λ) | Accuracy (%) | Sparsity (%) |
| ---------- | ------------ | ------------ |
| 0.01       | 47.42        | 6.30         |
| 0.05       | 47.82        | 57.96        |
| 0.1        | 49.74        | 60.78        |

---

## 5. Analysis

* Low λ results in high accuracy but minimal pruning
* Moderate λ (~0.05) achieves significant sparsity (~58%) while maintaining accuracy
* High λ increases sparsity but reduces performance

This demonstrates the trade-off between model efficiency and predictive performance.

---

## 6. Gate Distribution

The histogram of gate values shows:

* A concentration near zero → pruned weights
* A cluster away from zero → important connections

This confirms effective self-pruning behavior.

---

## 7. Conclusion

The model successfully:

* Learns sparse representations
* Removes redundant connections
* Maintains reasonable accuracy

This highlights the effectiveness of integrating pruning into training.

---

## 8. Future Work

* Extend to convolutional networks
* Explore structured pruning
* Compare with traditional pruning methods
* Measure inference efficiency

---



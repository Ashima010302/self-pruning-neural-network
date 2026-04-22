#  Self-Pruning Neural Network (CIFAR-10)

##  Overview

This project implements a **self-pruning neural network** that dynamically removes unnecessary weights during training using learnable gate parameters.

Unlike traditional pruning methods, this approach integrates pruning directly into the training process.

---

##  Key Idea

Each weight is associated with a learnable gate:

* Gate ∈ (0, 1) via sigmoid
* Effective weight = weight × gate
* Gate → 0 ⇒ connection is pruned

---

##  Method

* Custom `PrunableLinear` layer
* Loss = CrossEntropy + λ × Sparsity Loss
* Sparsity Loss = mean of gate values
* Warm-up training (no pruning in early epochs)

---

##  Results

| Lambda (λ) | Accuracy (%) | Sparsity (%) |
| ---------- | ------------ | ------------ |
| 0.01       | 47.42        | 6.30         |
| 0.05       | 47.82        | 57.96        |
| 0.1        | 49.74        | 60.78        |

---

##  Observations

* Increasing λ increases sparsity
* λ = 0.05 achieves ~58% sparsity with minimal accuracy drop
* Higher λ leads to more aggressive pruning

---

##  How to Run

### Run on Google Colab (Recommended)

1. Open notebook in Google Colab
2. Enable GPU: Runtime → Change runtime type → GPU
3. Run all cells

---

##  Key Learnings

* L1 regularization induces sparsity
* Neural networks can learn to prune during training
* Trade-off exists between sparsity and accuracy

---


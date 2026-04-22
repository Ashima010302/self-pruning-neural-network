# Self-Pruning Neural Network (CIFAR-10)

##  Overview

This project implements a **self-pruning neural network** that learns to remove unnecessary weights during training using learnable gate parameters.

Unlike traditional pruning techniques applied after training, this model **dynamically prunes itself during training**, making it more efficient and adaptive.

---

##  Key Idea

Each weight in the network is associated with a learnable **gate parameter**:

* Gate values ∈ (0, 1) via sigmoid
* Effective weight = `weight × gate`
* If gate → 0 → connection is pruned

This allows the network to **learn which connections are important**.

---

##  Methodology

###  Prunable Layer

A custom `PrunableLinear` layer is implemented:

* Learns both weights and gate scores
* Applies sigmoid transformation to gates
* Multiplies weights with gates before forward pass

---

###  Loss Function

Total Loss = Classification Loss + λ × Sparsity Loss

* **Classification Loss** → Cross-Entropy
* **Sparsity Loss** → Mean of gate values

This encourages the network to **reduce unnecessary connections**.

---

###  Training Strategy

A **warm-up phase** is used:

* Initial epochs → only classification loss
* Later epochs → classification + sparsity loss

This stabilizes training and prevents early pruning collapse.

---

##  Results

| Lambda (λ) | Accuracy (%)       | Sparsity (%) |
| ---------- | ------------------ | ------------ |
| 0.01       | 47.42              | 6.30         |
| 0.05       | 47.82              | 57.96        |
| 0.1        | 47.94              | 60.78        |

---

##  Key Observations

* **Low λ (0.01)**

  * High accuracy
  * Minimal pruning

* **Medium λ (0.05)** 

  * ~58% sparsity
  * Maintains accuracy
  * Best trade-off

* **High λ (0.1)**

  * Higher sparsity
  * Slight accuracy drop

 A clear **trade-off between sparsity and performance** is observed.

---

##  Gate Distribution

The distribution of gate values shows:

* Many values near **0** → pruned connections
* Some values away from 0 → important weights

This confirms successful **self-pruning behavior**.

---

## Tech Stack

* Python
* PyTorch
* NumPy
* Matplotlib

---


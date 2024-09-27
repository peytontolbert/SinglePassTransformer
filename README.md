# Single-Pass Training Method for Transformer Models

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Comparison](#comparison)
- [Single-Pass Training Method](#single-pass-training-method)
  - [Overview](#overview)
  - [Advantages](#advantages)
  - [Implementation Details](#implementation-details)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

## Introduction

Welcome to the **Single-Pass Training Method** project! This project introduces an innovative approach to training transformer-based models by leveraging a single-pass dataset loading mechanism. By optimizing the training process, our method enhances efficiency, reduces computational overhead, and maintains high performance across various tasks.

Mathematical Derivation for Closed-Form Solution
1. General Formulation
In traditional gradient descent, the weight update for each batch 
𝑖 can be expressed as:
```math
𝑊𝑡+1=𝑊𝑡−𝜂∇𝑊𝐿𝑖(𝑊𝑡)
```
Where:

𝑊𝑡  is the weight at step 𝑡.
𝜂 is the learning rate.
𝐿𝑖 is the loss function for batch 𝑖.


After one epoch, the cumulative weight update for all batches 
{𝑖=1,2,…,𝑁batches} can be represented as:

```math
𝑊new=𝑊old−𝜂  𝑖=1∑𝑁batches  ∇𝑊𝐿𝑖(𝑊old)
```
This summation accounts for the cumulative gradient descent updates over the entire epoch.

2. Incorporating Batch-Wise Gradient Effects
To derive a closed-form solution, we need to simulate the effect of multiple gradient updates for each batch in a single step. This involves computing an effective gradient that encapsulates the contributions of all batches:

```math
ΔW=−η i=1∑Nbatches  ∇W Li(W)
```

This equation is equivalent to computing a "global gradient" that captures the cumulative effect of all individual batch gradients.

3. Closed-Form Solution Using Kernel Matrix
To extend this concept to our kernel-based model, we use the kernel matrix 𝐾 that captures the relationships between data points. The optimal weight vector in the kernel space can be calculated using:
```math
𝛼=(𝐾+𝜆𝐼)−1𝑌
```
Where:

𝐾 is the kernel matrix representing the inner product of all training embeddings.
𝜆𝐼 is a regularization term.
𝑌 represents the target values.





## Features

- **Single-Pass Training:** Streamlined dataset loading for efficient training.
- **Transformer Integration:** Utilizes state-of-the-art transformer architectures.
- **Performance Optimization:** Reduced memory usage and faster convergence.
- **Modular Design:** Easily extendable components for customization.

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/peytontolbert/SinglePassTransformer.git
   cd SinglePassTransformer
   ```

2. **Create a Virtual Environment**
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

## Usage

### Training

To train the model using the single-pass training method:
```bash
python train.py --method single-pass --epochs 50 --batch_size 32 --learning_rate 1e-3
```

### Comparison

For benchmarking against traditional training methods, use the `compare.py` script:
```bash
python compare.py
```

This script runs both traditional and single-pass training methods, comparing metrics such as training time and memory usage.

## Single-Pass Training Method

### Overview

The single-pass training method revolutionizes the way datasets are loaded and processed during model training. Unlike traditional methods that may require multiple passes over the data, our approach ensures that each data batch is processed in a single pass, significantly enhancing training efficiency.

### Advantages

- **Efficiency:** Reduces the number of data loading operations, speeding up the training process.
- **Resource Optimization:** Lowers memory consumption by handling data more effectively.
- **Scalability:** Facilitates training on larger datasets without additional computational costs.
- **Simplified Pipeline:** Streamlines the training workflow, making it easier to integrate and manage.

### Implementation Details

Our single-pass training method is implemented in the `train.py` script. Key components include:

- **Dataset Loader:** Utilizes a single-pass data loading mechanism to feed data batches to the model.
- **Model Architecture:** Employs a simple transformer-based model (`SimpleTransformer`) designed for binary classification tasks.
- **Training Loop:** Optimizes the training loop to handle data in a single pass, updating model weights efficiently.
- **Evaluation:** Provides functions to evaluate model performance on new data.

```python:train.py
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
from datasets import load_dataset
```

Incorporate the batch-wise gradient effects analytically in the solution
```python
def compute_batch_aware_solution(K, y, num_epochs, batch_size, learning_rate, lambda_reg=1e-5):
n = K.shape[0]
I = torch.eye(n)
# Simulate the effect of batch-wise gradient updates
batch_effect = (n // batch_size) learning_rate num_epochs # Approximate cumulative effect over batches
effective_K = K + lambda_reg I
```

# Incorporate batch effects in the kernel matrix
```python
A_inv = torch.inverse(effective_K batch_effect + lambda_reg I)
alpha = torch.mm(A_inv, y)
return alpha
```

## Project Structure
```markdown
single-pass-training/
├── compare.py
├── traditional.py
├── train.py
├── README.md
```

- **compare.py:** Script to compare traditional and single-pass training methods.
- **traditional.py:** Implements the traditional training approach for baseline comparison.
- **train.py:** Main script using the single-pass training method.
- **README.md:** Project documentation.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your enhancements.

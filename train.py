import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
from datasets import load_dataset


# Load pre-trained model and tokenizer
model_name = 'gpt2'  # Choose 'gpt2', 'gpt2-medium', 'gpt2-large', etc.
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name, output_hidden_states=True)
tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token
# Example text data (extend this with your actual dataset)
texts = [
    "Once upon a time, in a land far away, there was a kingdom.",
    "The quick brown fox jumps over the lazy dog.",
    "In the beginning, there was nothing but darkness.",
    "She sells seashells by the seashore."
]


# Separate inputs and outputs
input_texts = [pair[0] for pair in texts]
output_texts = [pair[1] for pair in texts]

# Tokenization and embedding
def get_embeddings(texts):
    inputs = tokenizer(
        texts,
        padding='max_length',          # Enable dynamic padding
        truncation=True,       # Enable truncation
        max_length=128,        # Set maximum sequence length here
        return_tensors='pt'
    )
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.hidden_states[-1]
    attention_mask = inputs['attention_mask']
    masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
    sum_embeddings = masked_embeddings.sum(dim=1)
    lengths = attention_mask.sum(dim=1, keepdim=True)
    embeddings = sum_embeddings / lengths
    return embeddings

# Get embeddings for inputs and outputs
input_embeddings = get_embeddings(input_texts).float()
output_embeddings = get_embeddings(output_texts).float()

# Debug: Print embeddings shape
print("Input Embeddings shape:", input_embeddings.shape)
print("Output Embeddings shape:", output_embeddings.shape)

# Initialize the kernel with input_dim
input_dim = input_embeddings.shape[1]
print("Input dimension set to:", input_dim)

# Define a custom deep kernel function
class DeepKernel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DeepKernel, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x1, x2):
        x1 = self.activation(self.layer1(x1))
        x1 = self.activation(self.layer2(x1))
        x1 = x1 / x1.norm(dim=1, keepdim=True) # Normalize activations
        x2 = self.activation(self.layer1(x2))
        x2 = self.activation(self.layer2(x2))
        x2 = x2 / x2.norm(dim=1, keepdim=True) # Normalize activations
        return torch.mm(x1, x2.T)

# Initialize the kernel with fixed input_dim
hidden_dim = 128  # You can adjust this parameter
kernel = DeepKernel(input_dim, hidden_dim)

# Compute the kernel matrix
K = kernel(input_embeddings, input_embeddings)

# Debug: Print kernel matrix shape
print("Kernel matrix shape:", K.shape)


# Incorporate the batch-wise gradient effects analytically in the solution
def compute_batch_aware_solution(K, y, num_epochs, batch_size, learning_rate, lambda_reg=1e-5):
    n = K.shape[0]
    I = torch.eye(n)
    # Simulate the effect of batch-wise gradient updates
    batch_effect = (n // batch_size) * learning_rate * num_epochs # Approximate cumulative effect over batches
    effective_K = K + lambda_reg * I
    # Incorporate batch effects in the kernel matrix
    A_inv = torch.inverse(effective_K * batch_effect + lambda_reg * I)
    alpha = torch.mm(A_inv, y)
    return alpha

# Simulate the batch size and learning rate effects
batch_size = 2  # Set batch size (number of samples per batch)
learning_rate = 1e-3  # Set learning rate for simulation

# Number of iterations to simulate
num_epochs = 50  # Adjust this to simulate more iterations

# Compute alpha (weights)
alpha = compute_batch_aware_solution(K, output_embeddings, num_epochs, batch_size, learning_rate)

# Debug: Print alpha values
print("Alpha values:", alpha)

# Prediction function
def predict(new_texts, embeddings, alpha, kernel):
    new_embeddings = get_embeddings(new_texts).float()
    K_new = kernel(new_embeddings, embeddings)
    predictions = torch.mm(K_new, alpha)
    return predictions

# Example prediction
new_texts = ["A friendly dog.", "Cats are playful."]
predictions = predict(new_texts, output_embeddings, alpha, kernel)
print("Predictions:", predictions.detach().numpy())

# Debug: Print predictions
print("Predictions:", predictions.detach().numpy())

# Compare the predictions from both models
print("Single-Pass Model Predictions:", predictions.detach().numpy())

# Debug: Print single-pass model predictions
print("Single-Pass Model Predictions:", predictions.detach().numpy())
import torch
import torch.nn as nn
from transformers import BertTokenizer, GPT2Tokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset

# Initialize the tokenizer (using GPT-2 to match evolved.py)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token to match evolved.py

# Define a simple transformer model
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        return x

# Initialize the model
vocab_size = tokenizer.vocab_size
embed_dim = 128
num_classes = 1
model = SimpleTransformer(vocab_size, embed_dim, num_classes)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Use the same example text data as evolved.py
texts = [
    "Once upon a time, in a land far away, there was a kingdom.",
    "The quick brown fox jumps over the lazy dog.",
    "In the beginning, there was nothing but darkness.",
    "She sells seashells by the seashore."
]
targets = torch.tensor([[1], [0], [1], [0]]).float()  # Binary targets for demonstration

# Tokenization and embedding now using DataLoader for batching
def get_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
    embeddings = inputs['input_ids']
    return embeddings

# Create DataLoader
batch_size = 2
input_ids = get_embeddings(texts).long()
labels = targets
dataset = torch.utils.data.TensorDataset(input_ids, labels)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
num_epochs = 50  # Adjust as needed
for epoch in range(num_epochs):
    for step, (batch_input_ids, batch_labels) in enumerate(loader):
        optimizer.zero_grad()
        outputs = model(batch_input_ids)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}], Loss: {loss.item():.4f}")

# Evaluation function
def evaluate(model, texts):
    model.eval()
    with torch.no_grad():
        inputs = get_embeddings(texts).long()
        outputs = model(inputs)
        predictions = torch.sigmoid(outputs)
    return predictions

# Example prediction
new_texts = ["A friendly dog.", "Cats are playful."]

# Evaluate on the same new texts
traditional_predictions = evaluate(model, new_texts)
print("Traditional Model Predictions:", traditional_predictions.detach().numpy())
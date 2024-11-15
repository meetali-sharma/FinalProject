# -*- coding: utf-8 -*-
"""Seq2Seq_Project.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/19RSxasr7GPBgIr_9q_guJuXd8Ne5eXyS
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generate dummy(Synthetic) data
def generate_data(num_samples, seq_len, vocab_size):
    data = []
    for _ in range(num_samples):
        src = [random.randint(1, vocab_size-1) for _ in range(seq_len)]   # Define the source sequence as a randomly generated sequence of integers.
        tgt = src[::-1]       # Define the target sequence as a reverse of source sequence.
        data.append((src, tgt))
    return data

# Initialize parameters for data generation
vocab_size = 20
seq_len = 10
num_samples = 1000

# Generate synthetic data
data = generate_data(num_samples, seq_len, vocab_size)

from torch.utils.data import Dataset, DataLoader

# Sequence-to-Sequence Model with Attention Mechanism

#Custom Dataset for Seq2Seq tasks
class Seq2SeqDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

# Initialize dataset and dataloader
dataset = Seq2SeqDataset(data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Attention Module
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(1)
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)
        attn_energies = self.score(H, encoder_outputs)
        return torch.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        energy = torch.bmm(v, energy)
        return energy.squeeze(1)

# Encoder Module
class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        embedding = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedding)
        return outputs, hidden, cell

# Decoder Module with Attention
class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, num_layers, attention):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.attention = attention
        self.embedding = nn.Embedding(output_size, embed_size)
        self.lstm = nn.LSTM(hidden_size + embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell, encoder_outputs):
        x = x.unsqueeze(1)
        embedding = self.embedding(x)
        attn_weights = self.attention(hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs)
        rnn_input = torch.cat((embedding, context), dim=2)
        outputs, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        predictions = self.fc(outputs.squeeze(1))
        return predictions, hidden, cell

# Seq2Seq Model with Attention Mechanism
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.decoder.output_size

        # Initialize a tensor to store decoder outputs
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(device)

        # Pass the source through the encoder
        encoder_outputs, hidden, cell = self.encoder(source)

        # First input to the decoder is the start token
        x = target[:, 0]

        for t in range(1, target_len):
            # Forward pass through the decoder
            output, hidden, cell = self.decoder(x, hidden, cell, encoder_outputs)
            outputs[:, t, :] = output

            # Decide whether to use teacher forcing
            best_guess = output.argmax(1)
            x = target[:, t] if random.random() < teacher_forcing_ratio else best_guess

        return outputs

# Initialize the model(Model Hyperparameters)
input_size = vocab_size
output_size = vocab_size
embed_size = 256
hidden_size = 512
num_layers = 2

# Initialize components
attention = Attention(hidden_size)
encoder = Encoder(input_size, embed_size, hidden_size, num_layers).to(device)
decoder = Decoder(output_size, embed_size, hidden_size, num_layers, attention).to(device)
model = Seq2Seq(encoder, decoder).to(device)

# # Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model

num_epochs = 10
loss_history = [] # Initialize loss_history outside the loop

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0 # Initialize epoch_loss for each epoch

    for i, (src, tgt) in enumerate(dataloader):
        src, tgt = src.to(device), tgt.to(device)

        optimizer.zero_grad()
        output = model(src, tgt)

        output = output[:, 1:].reshape(-1, output.shape[2])
        tgt = tgt[:, 1:].reshape(-1)

        loss = criterion(output, tgt)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()            # Accumulate loss for the epoch

    avg_loss = epoch_loss / len(dataloader)  # Calculate average loss for the epoch
    loss_history.append(avg_loss)            # Append average loss to history

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# Plot the training loss over epochs
if epoch > 0: # Start plotting from the second epoch
    plt.plot(range(1, epoch + 1), loss_history[:-1], marker='o') # Exclude current epoch's loss
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.show()

# Evaluating the model

model.eval()

all_targets = []
all_preds = []

with torch.no_grad():
    for i, (src, tgt) in enumerate(dataloader):
        src, tgt = src.to(device), tgt.to(device)

        output = model(src, tgt, teacher_forcing_ratio=0)
        output = output[:, 1:].reshape(-1, output.shape[2])
        tgt = tgt[:, 1:].reshape(-1)

        pred = output.argmax(1) # Get predicted values for each token

        all_targets.extend(tgt.cpu().numpy())  # Extend with individual target values
        all_preds.extend(pred.cpu().numpy())   # Extend with individual predicted values

        predicted = output.argmax(1).view(-1, seq_len-1)
        print(f'Source: {src[0].cpu().numpy()}')
        print(f'Target: {tgt.view(-1, seq_len-1)[0].cpu().numpy()}')
        print(f'Predicted: {predicted[0].cpu().numpy()}')
        break

# Calculate overall metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='macro')
    recall = recall_score(all_targets, all_preds, average='macro')
    f1 = f1_score(all_targets, all_preds, average='macro')

    print("Classification Report:")
    print(classification_report(all_targets, all_preds))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")


# Metrics to plot
metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1,
}

# Bar plot
plt.figure(figsize=(10, 6))
plt.bar(metrics.keys(), metrics.values(), color=['skyblue', 'orange', 'green', 'red'])
plt.ylim(0, 1)  # Assuming metrics are in [0, 1]
plt.xlabel("Metric")
plt.ylabel("Score")
plt.title("Seq2Seq Model Performance Metrics")
plt.xticks(rotation=45)
plt.show()
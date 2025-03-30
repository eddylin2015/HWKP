import torch
import torch.nn as nn
import torch.optim as optim

# Load a larger dataset: Example with Shakespeare text
with open("shakespeare.txt", "r") as file:
    corpus = file.read().lower()

chars = list(set(corpus))  # Unique characters
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Hyperparameters
seq_length = 20  # Longer sequences for more context
hidden_size = 256  # More hidden units
num_layers = 2  # Multi-layer LSTM
learning_rate = 0.005
epochs = 300

# Prepare dataset
def prepare_data(corpus, seq_length):
    inputs, targets = [], []
    for i in range(len(corpus) - seq_length):
        inputs.append(corpus[i:i+seq_length])
        targets.append(corpus[i+seq_length])
    return inputs, targets

inputs, targets = prepare_data(corpus, seq_length)
x = torch.tensor([[char_to_idx[ch] for ch in seq] for seq in inputs], dtype=torch.long)
y = torch.tensor([char_to_idx[ch] for ch in targets], dtype=torch.long)

# Define the model
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(LSTMLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])  # Take the last output
        return out, hidden

model = LSTMLanguageModel(vocab_size, hidden_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(epochs):
    hidden = None
    optimizer.zero_grad()
    output, hidden = model(x, hidden)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')

# Generate text
def generate_text(model, start_seq, length):
    model.eval()
    input_seq = torch.tensor([[char_to_idx[ch] for ch in start_seq]])
    generated = start_seq
    hidden = None
    for _ in range(length):
        with torch.no_grad():
            output, hidden = model(input_seq, hidden)
            pred_idx = torch.argmax(output, dim=1).item()
            generated += idx_to_char[pred_idx]
            input_seq = torch.tensor([[pred_idx]])
    return generated

# Generate text using the trained model
start_sequence = "to be or not to be"
print(generate_text(model, start_sequence, 20))

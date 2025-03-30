import torch
import torch.nn as nn
import torch.optim as optim

# Example corpus and tokenization
corpus = "hello world! hello ai!"
chars = list(set(corpus))  # Unique characters
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Hyperparameters
seq_length = 4
hidden_size = 128
num_layers = 1
lr = 0.01
epochs = 100

# Prepare data
def prepare_data(corpus, seq_length):
    inputs = []
    targets = []
    for i in range(len(corpus) - seq_length):
        inputs.append(corpus[i:i+seq_length])
        targets.append(corpus[i+seq_length])
    return inputs, targets

inputs, targets = prepare_data(corpus, seq_length)
x = torch.tensor([[char_to_idx[ch] for ch in seq] for seq in inputs])
y = torch.tensor([char_to_idx[ch] for ch in targets])

# Define model
class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(TinyLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

model = TinyLanguageModel(vocab_size, hidden_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training
for epoch in range(epochs):
    hidden = None
    optimizer.zero_grad()
    output, hidden = model(x, hidden)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Generate text
def generate_text(model, start_seq, length):
    model.eval()
    hidden = None
    input_seq = torch.tensor([[char_to_idx[ch] for ch in start_seq]])
    generated = start_seq
    for _ in range(length):
        with torch.no_grad():
            output, hidden = model(input_seq, hidden)
            pred_idx = torch.argmax(output, dim=1).item()
            generated += idx_to_char[pred_idx]
            input_seq = torch.tensor([[pred_idx]])
    return generated

start_sequence = "hello"
print(generate_text(model, start_sequence, 20))

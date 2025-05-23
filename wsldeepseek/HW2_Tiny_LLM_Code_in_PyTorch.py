
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Hyperparameters
vocab_size = 10000  # Size of the vocabulary
embed_dim = 128     # Embedding dimension
num_heads = 4       # Number of attention heads
num_layers = 2      # Number of Transformer layers
seq_length = 32     # Sequence length
batch_size = 16     # Batch size
learning_rate = 1e-4
epochs = 5          # Number of training epochs

# Tiny Transformer-based Language Model
class TinyLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super(TinyLLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_length, embed_dim))
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x) + self.positional_encoding
        for block in self.transformer_blocks:
            x = block(x)
        return self.fc_out(x)

# Dummy Dataset (Replace with your own dataset)
class TextDataset(Dataset):
    def __init__(self, text, seq_length):
        self.text = text
        self.seq_length = seq_length
        self.tokenized_text = torch.randint(0, vocab_size, (len(text),))

    def __len__(self):
        return len(self.text) - self.seq_length

    def __getitem__(self, idx):
        x = self.tokenized_text[idx:idx + self.seq_length]
        y = self.tokenized_text[idx + 1:idx + self.seq_length + 1]
        return x, y

# Prepare Data
text = "This is a tiny language model implemented in PyTorch. " * 1000  # Dummy text
dataset = TextDataset(text, seq_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize Model, Loss, and Optimizer
model = TinyLLM(vocab_size, embed_dim, num_heads, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

    print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {total_loss / len(dataloader):.4f}")

# Save the Model
torch.save(model.state_dict(), "tiny_llm.pth")

# Generate Text (Inference)
def generate_text(model, start_text, max_length=50):
    model.eval()
    input_ids = torch.randint(0, vocab_size, (1, seq_length))  # Replace with tokenizer in real use
    generated_text = []
    for _ in range(max_length):
        with torch.no_grad():
            output = model(input_ids)
            next_token = output.argmax(dim=-1)[:, -1].item()
            generated_text.append(next_token)
            input_ids = torch.cat([input_ids[:, 1:], torch.tensor([[next_token]])], dim=1)
    return generated_text

# Load the Model and Generate Text
model.load_state_dict(torch.load("tiny_llm.pth"))
generated_text = generate_text(model, start_text="This is")
print("Generated Text:", generated_text)
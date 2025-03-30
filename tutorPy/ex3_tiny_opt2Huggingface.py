# basic setup
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
# define a tiny transformer
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.fc_out(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1)]
        return x

# create a simple dataset        
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_length=32):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.data = self.process_texts(texts)
        
    def process_texts(self, texts):
        tokenized = [self.tokenizer(text) for text in texts]
        return torch.cat(tokenized)
        
    def __len__(self):
        return len(self.data) - self.seq_length
        
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_length]
        y = self.data[idx+1:idx+self.seq_length+1]
        return x, y

# Simple character-level tokenizer
class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char2idx = {c:i for i,c in enumerate(self.chars)}
        self.idx2char = {i:c for i,c in enumerate(self.chars)}
        
    def __call__(self, text):
        return torch.tensor([self.char2idx[c] for c in text], dtype=torch.long)

#Create a Simple Dataset

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_length=32):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.data = self.process_texts(texts)
        
    def process_texts(self, texts):
        tokenized = [self.tokenizer(text) for text in texts]
        return torch.cat(tokenized)
        
    def __len__(self):
        return len(self.data) - self.seq_length
        
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_length]
        y = self.data[idx+1:idx+self.seq_length+1]
        return x, y

# Simple character-level tokenizer
class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char2idx = {c:i for i,c in enumerate(self.chars)}
        self.idx2char = {i:c for i,c in enumerate(self.chars)}
        
    def __call__(self, text):
        return torch.tensor([self.char2idx[c] for c in text], dtype=torch.long)
        
#Training Loop

def train_tiny_lm():
    # Sample text
    text = "Hello world! This is a tiny language model. " * 100
    
    # Initialize tokenizer and model
    tokenizer = CharTokenizer(text)
    model = TinyTransformer(tokenizer.vocab_size, d_model=64, nhead=2, num_layers=2)
    
    # Create dataset and dataloader
    dataset = TextDataset([text], tokenizer, seq_length=32)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Training loop
    for epoch in range(10):
        model.train()
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output.view(-1, tokenizer.vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
    
    return model, tokenizer
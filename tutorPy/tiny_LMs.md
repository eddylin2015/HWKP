# Building Tiny Language Models on Google Colab

Creating small language models (tiny LMs) is a great way to learn about transformer architectures while working within Colab's hardware constraints. Here's a comprehensive guide to building and training your own tiny LM in Colab.

## Option 1: Building from Scratch with PyTorch

### 1. Basic Setup

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
```

### 2. Define a Tiny Transformer

```python
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
```

### 3. Create a Simple Dataset

```python
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
```

### 4. Training Loop

```python
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
```

## Option 2: Using Hugging Face's Transformers

For a more production-ready but still small model:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

def train_tiny_gpt():
    # Load a small pre-trained model
    model_name = "distilgpt2"  # Much smaller than GPT-2
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    
    # Prepare tiny dataset
    texts = ["This is a tiny language model example. "] * 100
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=32)
    dataset = torch.utils.data.TensorDataset(inputs['input_ids'], inputs['attention_mask'])
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./tiny_model",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=100,
        save_total_limit=2,
        logging_steps=10,
        fp16=True,  # Use mixed precision
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    # Train
    trainer.train()
    return model, tokenizer
```

## Option 3: NanoGPT Implementation

A simplified version of Andrej Karpathy's nanoGPT:

```python
class NanoGPT(nn.Module):
    def __init__(self, vocab_size, block_size=128, n_embd=64, n_head=4, n_layer=4):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, x):
        B, T = x.shape
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(torch.arange(T, device=x.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(x)
```

## Tips for Tiny LM Training in Colab

1. **Start Small**:
   - Embedding dimensions: 64-128
   - Layers: 2-4
   - Heads: 2-4
   - Sequence length: 64-128

2. **Monitor Memory**:
   ```python
   !nvidia-smi
   ```

3. **Use Mixed Precision**:
   ```python
   scaler = torch.cuda.amp.GradScaler()
   with torch.cuda.amp.autocast():
       outputs = model(inputs)
   ```

4. **Save Checkpoints**:
   ```python
   torch.save({
       'model_state_dict': model.state_dict(),
       'tokenizer': tokenizer,
   }, 'tiny_model.pth')
   ```

5. **Test Generation**:
   ```python
   def generate_text(model, tokenizer, prompt, max_length=20):
       model.eval()
       input_ids = tokenizer(prompt).unsqueeze(0).to(device)
       with torch.no_grad():
           for _ in range(max_length):
               outputs = model(input_ids)
               next_token = outputs[0, -1].argmax()
               input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
       return tokenizer.decode(input_ids[0].tolist())
   ```

Would you like me to focus on any particular aspect (character-level vs token-level, specific architecture details, or training techniques)?
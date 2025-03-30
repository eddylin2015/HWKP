# Training Large Language Models (LLMs) in Google Colab

Google Colab can be a useful platform for training smaller LLMs or fine-tuning existing models, though it has limitations for very large models due to hardware constraints. Here's a guide to getting started:

## Hardware Considerations in Colab

- **Free Tier**: NVIDIA T4 GPU (16GB VRAM) or occasionally K80 (12GB VRAM)
- **Pro/Pro+**: Access to A100 (40GB VRAM) or V100 (16GB/32GB VRAM)
- **RAM**: Typically 12GB-25GB system RAM
- **Storage**: ~100GB temporary disk space

## Approaches for LLM Training in Colab

### 1. Fine-tuning Existing Models

```python
# Example for fine-tuning a Hugging Face model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

model_name = "gpt2"  # or other smaller models like distilgpt2
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Prepare your dataset (format depends on your task)
train_dataset = ... 
eval_dataset = ...

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=500,
    evaluation_strategy="steps",
    eval_steps=2000,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

### 2. LoRA (Low-Rank Adaptation)

For more efficient fine-tuning:

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Should show much fewer parameters
```

### 3. Training from Scratch (Small Models)

```python
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

# Initialize a small model
config = GPT2Config(
    vocab_size=50257,
    n_positions=1024,
    n_embd=256,
    n_layer=6,
    n_head=8
)
model = GPT2LMHeadModel(config)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Training loop would go here
```

## Tips for Colab LLM Training

1. **Use Mixed Precision**: 
   ```python
   training_args = TrainingArguments(fp16=True, ...)
   ```

2. **Gradient Checkpointing**:
   ```python
   model.gradient_checkpointing_enable()
   ```

3. **Batch Size**: Start small (2-4) and increase if memory allows

4. **Monitor Resources**:
   ```python
   !nvidia-smi
   !free -h
   ```

5. **Save to Google Drive**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   # Then use paths like '/content/drive/MyDrive/model_checkpoints'
   ```

## Limitations

- Colab sessions time out after 12 hours (free) or 24 hours (Pro)
- VRAM limits model size (typically <1B parameters for training)
- Disk space is temporary unless saved to Drive

For larger models, consider:
- Using Colab just for inference
- Cloud services with better GPUs (AWS, GCP, Lambda Labs)
- Training on multiple Colab instances with model parallelism (advanced)

Would you like me to elaborate on any specific aspect of LLM training in Colab?
import torch
import numpy as np
from tqdm import tqdm
import os
from model import GPT
from config.train_shakespeare import config

# Simple character-level tokenizer
class CharTokenizer:
    def __init__(self, data):
        self.chars = sorted(list(set(data)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
    
    def encode(self, text):
        return [self.stoi[c] for c in text]
    
    def decode(self, tokens):
        return ''.join([self.itos[i] for i in tokens])

# Load data
def load_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

# Load tokenized binary files
def get_batch(split, block_size, batch_size, device):
    data = torch.tensor(np.fromfile(f'data/{split}.bin', dtype=np.uint16), dtype=torch.long).to(device)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device
    config.vocab_size = 65  # Will be updated after tokenizer initialization

    # Load and prepare data
    text = load_data('data/input.txt')
    tokenizer = CharTokenizer(text)
    config.vocab_size = tokenizer.vocab_size

    # Initialize model
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Training loop
    for iter in tqdm(range(config.max_iters)):
        xb, yb = get_batch('train', config.block_size, config.batch_size, device)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter % config.eval_interval == 0:
            print(f"Iter {iter}: Loss {loss.item():.4f}")

        if iter % config.save_interval == 0 and iter > 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f'checkpoints/model_iter_{iter}.pt')

    # Save final model
    torch.save(model.state_dict(), 'checkpoints/model_final.pt')
    print("Training completed. Model saved to checkpoints/model_final.pt")

if __name__ == "__main__":
    main()
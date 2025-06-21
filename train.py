import torch
import numpy as np
from tqdm import tqdm
import os
import json
from model import GPT
from config.train_shakespeare import config

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from torch.amp import GradScaler
from torch.amp import autocast

class CharTokenizer:
    def __init__(self, data=None, vocab_file=None):
        if vocab_file:
            with open(vocab_file, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            self.stoi = vocab_data['stoi']
            self.itos = {int(k): v for k, v in vocab_data['itos'].items()}
            self.chars = sorted(list(self.stoi.keys()))
            self.vocab_size = len(self.chars)
        elif data:
            self.chars = sorted(list(set(data)))
            self.vocab_size = len(self.chars)
            self.stoi = {ch: i for i, ch in enumerate(self.chars)}
            self.itos = {i: ch for i, ch in enumerate(self.chars)}
        else:
            raise ValueError("Either 'data' or 'vocab_file' must be provided.")

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, tokens):
        return ''.join([self.itos[i] for i in tokens])

def get_batch(data_tensor, block_size, batch_size, device):
    ix = torch.randint(len(data_tensor) - block_size, (batch_size,))
    x = torch.stack([data_tensor[i:i+block_size] for i in ix])
    y = torch.stack([data_tensor[i+1:i+block_size+1] for i in ix])
    x = x.to(device)
    y = y.to(device)
    return x, y

def _train_loop_fn(rank, config_obj):
    device = xm.xla_device()
    config_obj.device = device

    tokenizer = CharTokenizer(vocab_file='data/vocab.json')
    config_obj.vocab_size = tokenizer.vocab_size

    train_data_full = torch.tensor(np.fromfile('data/train.bin', dtype=np.uint16), dtype=torch.long)
    val_data_full = torch.tensor(np.fromfile('data/val.bin', dtype=np.uint16), dtype=torch.long)

    model = GPT(config_obj).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config_obj.learning_rate)
    scaler = GradScaler(device=device)

    for iter_step in tqdm(range(config_obj.max_iters), desc=f"Training (Rank {rank})"):
        xb, yb = get_batch(train_data_full, config_obj.block_size, config_obj.batch_size, device)
        with autocast(device_type=device.type):
            logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if iter_step % config_obj.eval_interval == 0:
            if xm.is_master_ordinal():
                print(f"Iter {iter_step}: Training Loss {loss.item():.4f}")

        if iter_step % config_obj.save_interval == 0 and iter_step > 0:
            if xm.is_master_ordinal():
                os.makedirs('checkpoints', exist_ok=True)
                xm.save(model.state_dict(), f'checkpoints/model_iter_{iter_step}.pt')
                xm.mark_step()

    if xm.is_master_ordinal():
        os.makedirs('checkpoints', exist_ok=True)
        xm.save(model.state_dict(), 'checkpoints/model_final.pt')
        print("Training completed. Model saved to checkpoints/model_final.pt")

def main():
    xmp.spawn(_train_loop_fn, args=(config,)) 

if __name__ == "__main__":
    main()

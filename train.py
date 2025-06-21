import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import numpy as np
from tqdm import tqdm
import os
import json
from model import GPT
from config.train_shakespeare import config

class CharTokenizer:
    def __init__(self, data=None, vocab_file=None):
        if vocab_file:
            with open(vocab_file, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            self.stoi = vocab_data['stoi']
            self.itos = {int(k): v for k, v in vocab_data['itos'].items()}
            self.chars = sorted(list(self.stoi.keys()))
            if '<UNK>' not in self.stoi:
                self.stoi['<UNK>'] = len(self.stoi)
                self.itos[len(self.stoi) - 1] = '<UNK>'
            self.vocab_size = len(self.stoi)
        elif data:
            self.chars = sorted(list(set(data)))
            self.vocab_size = len(self.chars) + 1
            self.stoi = {ch: i for i, ch in enumerate(self.chars)}
            self.stoi['<UNK>'] = len(self.chars)
            self.itos = {i: ch for i, ch in enumerate(self.chars)}
            self.itos[len(self.chars)] = '<UNK>'
        else:
            raise ValueError("Either 'data' or 'vocab_file' must be provided.")

    def encode(self, text):
        return [self.stoi.get(c, self.stoi['<UNK>']) for c in text]

    def decode(self, tokens):
        return ''.join([self.itos.get(i, '<UNK>') for i in tokens])

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
    xm.rendezvous('init')  # Синхронизация процессов
    print(f"Rank {rank}: Device: {device}")

    tokenizer = CharTokenizer(vocab_file='data/vocab.json')
    config_obj.vocab_size = tokenizer.vocab_size
    print(f"Rank {rank}: Tokenizer vocab_size: {tokenizer.vocab_size}")

    train_data_full = torch.tensor(np.fromfile('data/train.bin', dtype=np.uint16), dtype=torch.long).to(device)
    val_data_full = torch.tensor(np.fromfile('data/val.bin', dtype=np.uint16), dtype=torch.long).to(device)

    model = GPT(config_obj).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config_obj.learning_rate)

    for iter_step in tqdm(range(config_obj.max_iters), desc=f"Training (Rank {rank})"):
        xb, yb = get_batch(train_data_full, config_obj.block_size, config_obj.batch_size, device)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        xm.optimizer_step(optimizer)
        xm.mark_step()  # Синхронизация TPU

        if iter_step % config_obj.eval_interval == 0 and xm.is_master_ordinal():
            print(f"Rank {rank}, Iter {iter_step}: Training Loss {loss.item():.4f}")

        if iter_step % config_obj.save_interval == 0 and iter_step > 0 and xm.is_master_ordinal():
            os.makedirs('checkpoints', exist_ok=True)
            xm.save(model.state_dict(), f'checkpoints/model_iter_{iter_step}.pt')

    if xm.is_master_ordinal():
        os.makedirs('checkpoints', exist_ok=True)
        xm.save(model.state_dict(), 'checkpoints/model_final.pt')
        print("Training completed. Model saved to checkpoints/model_final.pt")

def main():
    xmp.spawn(_train_loop_fn, args=(config,))

if __name__ == "__main__":
    main()

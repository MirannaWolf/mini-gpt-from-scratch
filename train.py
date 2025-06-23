import torch
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

            # Приводим ключи itos к int
            self.itos = {int(k): v for k, v in vocab_data['itos'].items()}

            # Убедимся, что <UNK> есть в обоих словарях
            if '<UNK>' not in self.stoi:
                unk_id = len(self.stoi)
                self.stoi['<UNK>'] = unk_id
                self.itos[unk_id] = '<UNK>'

            self.vocab_size = len(self.stoi)

        elif data:
            self.chars = sorted(set(data))
            self.vocab_size = len(self.chars) + 1  # для <UNK>

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

def evaluate_loss(data_tensor, model, block_size, batch_size, device):
    model.eval()
    with torch.no_grad():
        xb, yb = get_batch(data_tensor, block_size, batch_size, device)
        _, loss = model(xb, yb)
    model.train()
    return loss.item()

def train_loop(config_obj):
    device = torch.device('cpu')
    config_obj.device = device
    torch.set_num_threads(4)  # Установите число ядер вашего CPU
    print(f"Device: {device}")

    # Проверка наличия файлов
    for file_path in ['data/vocab.json', 'data/train.bin', 'data/val.bin']:
        if not os.path.exists(file_path):
            print(f"Ошибка: Файл {file_path} не найден. Запустите data/prepare.py.")
            return

    tokenizer = CharTokenizer(vocab_file='data/vocab.json')
    config_obj.vocab_size = tokenizer.vocab_size
    print(f"Tokenizer vocab_size: {tokenizer.vocab_size}")

    train_data_full = torch.tensor(np.fromfile('data/train.bin', dtype=np.uint16), dtype=torch.long).to(device)
    val_data_full = torch.tensor(np.fromfile('data/val.bin', dtype=np.uint16), dtype=torch.long).to(device)

    model = GPT(config_obj).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config_obj.learning_rate)

    for iter_step in tqdm(range(config_obj.max_iters), desc="Training"):
        xb, yb = get_batch(train_data_full, config_obj.block_size, config_obj.batch_size, device)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter_step % config_obj.eval_interval == 0:
            val_loss = evaluate_loss(val_data_full, model, config_obj.block_size, config_obj.batch_size, device)
            print(f"Iter {iter_step}: Training Loss {loss.item():.4f}, Validation Loss {val_loss:.4f}")

        if iter_step % config_obj.save_interval == 0 and iter_step > 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f'checkpoints/model_iter_{iter_step}.pt')

    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/model_final.pt')
    print("Training completed. Model saved to checkpoints/model_final.pt")

def main():
    train_loop(config)

if __name__ == "__main__":
    main()
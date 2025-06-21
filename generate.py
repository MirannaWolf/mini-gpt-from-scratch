import torch
import torch_xla.core.xla_model as xm
import json
import os
from model import GPT
from config.train_shakespeare import config
from train import CharTokenizer

def load_tokenizer(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    tokenizer = CharTokenizer(vocab_file=vocab_path)
    return tokenizer

def main():
    device = xm.xla_device()
    print(f"Device: {device}")

    vocab_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data/vocab.json'))
    if not os.path.exists(vocab_path):
        print(f"Ошибка: Файл {vocab_path} не найден. Запустите data/prepare.py.")
        return
    tokenizer = load_tokenizer(vocab_path)
    config.vocab_size = tokenizer.vocab_size
    print(f"Tokenizer vocab_size: {tokenizer.vocab_size}")

    model = GPT(config).to(device)
    checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints/model_final.pt'))
    if not os.path.exists(checkpoint_path):
        print(f"Ошибка: Файл чекпоинта {checkpoint_path} не найден. Запустите train.py.")
        return
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    prompt = "My youngest boy"
    max_length = 100
    temperature = 0.8
    top_k = 50
    do_sample = True

    try:
        generated_text = model.generate_text(
            tokenizer=tokenizer,
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
            device=device
        )
        print("\nСгенерированный текст:")
        print(generated_text)
    except Exception as e:
        print(f"Ошибка при генерации текста: {str(e)}")

if __name__ == "__main__":
    main()

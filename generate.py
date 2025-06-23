import torch
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
    device = torch.device('cpu')
    print(f"Device: {device}")

    # Загрузка токенизатора
    vocab_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data/vocab.json'))
    if not os.path.exists(vocab_path):
        print(f"Ошибка: Файл {vocab_path} не найден. Запустите data/prepare.py.")
        return
    tokenizer = load_tokenizer(vocab_path)
    config.vocab_size = tokenizer.vocab_size
    print(f"Tokenizer vocab_size: {tokenizer.vocab_size}")

    # Инициализация модели
    model = GPT(config).to(device)

    # Загрузка чекпоинта
    checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints/model_final.pt'))
    if not os.path.exists(checkpoint_path):
        print(f"Ошибка: Файл чекпоинта {checkpoint_path} не найден. Запустите train.py.")
        return

    # ✅ Загрузка параметров с допущением несовпадения ключей
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)

    model.eval()

    # Параметры генерации
    prompt = "My youngest boy"
    max_length = 150
    temperature = 0.7
    top_k = 20
    do_sample = False

    try:
        # Преобразование prompt в токены
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)[None, :]  # shape: (1, T)

        for _ in range(max_length):
            if input_tensor.size(1) > config.block_size:
                input_tensor = input_tensor[:, -config.block_size:]

            logits, _ = model(input_tensor)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                top_k_values, _ = torch.topk(logits, top_k)
                min_topk = top_k_values[:, -1].unsqueeze(1)
                logits = torch.where(logits < min_topk, torch.full_like(logits, -float('Inf')), logits)

            probs = torch.nn.functional.softmax(logits, dim=-1)
            if do_sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                _, next_token = torch.topk(probs, k=1, dim=-1)

            input_tensor = torch.cat((input_tensor, next_token), dim=1)

        output_ids = input_tensor[0].tolist()
        generated_text = tokenizer.decode(output_ids)
        print("\nСгенерированный текст:")
        print(generated_text)

    except Exception as e:
        print(f"Ошибка при генерации текста: {str(e)}")

if __name__ == "__main__":
    main()

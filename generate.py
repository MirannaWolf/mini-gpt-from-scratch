import torch
import json
import os
from model import GPT
from config.train_shakespeare import config
from train import CharTokenizer

def load_tokenizer(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    tokenizer = CharTokenizer("")
    tokenizer.stoi = vocab['stoi']
    tokenizer.itos = {int(k): v for k, v in vocab['itos'].items()}
    # Добавляем <UNK> для неизвестных символов
    if '<UNK>' not in tokenizer.stoi:
        tokenizer.stoi['<UNK>'] = len(tokenizer.stoi)
        tokenizer.itos[len(tokenizer.stoi) - 1] = '<UNK>'
    tokenizer.vocab_size = len(tokenizer.stoi)
    return tokenizer

def main():
    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, CUDA available: {torch.cuda.is_available()}")

    # Загрузка токенизатора
    vocab_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data/vocab.json'))
    if not os.path.exists(vocab_path):
        print(f"Ошибка: Файл {vocab_path} не найден. Запустите data/prepare.py.")
        return
    tokenizer = load_tokenizer(vocab_path)
    config.vocab_size = tokenizer.vocab_size
    print(f"Tokenizer vocab_size: {tokenizer.vocab_size}")

    # Загрузка модели
    model = GPT(config).to(device)
    checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints/model_final.pt'))
    if not os.path.exists(checkpoint_path):
        print(f"Ошибка: Файл чекпоинта {checkpoint_path} не найден. Запустите train.py.")
        return
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Консольный интерфейс
    print("Генерация текста с GPT. Введите 'выход' для завершения.")
    while True:
        prompt = input("\nВведите начальный текст: ")
        if prompt.lower() == 'выход':
            break
        if not prompt.strip():
            print("Ошибка: Введите непустой начальный текст.")
            continue

        try:
            max_length = int(input("Максимальная длина текста (по умолчанию 50): ") or 50)
            temperature = float(input("Температура (0.1–1.5, по умолчанию 0.8): ") or 0.8)
            top_k = int(input("Top-k (10–100, по умолчанию 50): ") or 50)
            do_sample = input("Использовать сэмплирование? (да/нет, по умолчанию да): ").lower() in ['да', 'yes', '']

            # Проверка параметров
            if temperature < 0.1 or temperature > 1.5:
                print("Ошибка: Температура должна быть в диапазоне 0.1–1.5.")
                continue
            if top_k < 10 or top_k > 100:
                print("Ошибка: Top-k должен быть в диапазоне 10–100.")
                continue

            # Кодируем prompt с обработкой неизвестных символов
            input_ids = [tokenizer.stoi.get(c, tokenizer.stoi['<UNK>']) for c in prompt]
            print(f"Input IDs: {input_ids}, Max ID: {max(input_ids) if input_ids else 0}")

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

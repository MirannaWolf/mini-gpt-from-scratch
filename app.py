import torch
import gradio as gr
import os
import json
from model import GPT
from config.train_shakespeare import config
from train import CharTokenizer

def load_tokenizer(vocab_path):
    tokenizer = CharTokenizer(vocab_file=vocab_path)
    return tokenizer

def load_model_and_tokenizer():
    device = torch.device('cpu')
    config.device = device

    # Проверка путей к файлам
    vocab_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data/vocab.json'))
    checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints/model_final.pt'))

    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Файл {vocab_path} не найден. Запустите data/prepare.py.")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Файл чекпоинта {checkpoint_path} не найден. Запустите train.py.")

    # Загрузка токенизатора
    tokenizer = load_tokenizer(vocab_path)
    config.vocab_size = tokenizer.vocab_size

    # Загрузка модели
    model = GPT(config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    return model, tokenizer, device

def generate_text(prompt, max_length, temperature, top_k, do_sample=True):
    model, tokenizer, device = load_model_and_tokenizer()
    try:
        with torch.no_grad():
            generated_text = model.generate_text(
                tokenizer=tokenizer,
                prompt=prompt,
                max_length=int(max_length),
                temperature=float(temperature),
                top_k=int(top_k),
                do_sample=do_sample,
                device=device
            )
        return generated_text
    except Exception as e:
        return f"Ошибка при генерации текста: {str(e)}"

# Gradio интерфейс
iface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Промпт", placeholder="Введите начальный текст, например, 'My youngest boy'"),
        gr.Slider(minimum=10, maximum=200, value=100, step=1, label="Максимальная длина"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.8, step=0.1, label="Температура"),
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Top-k")
    ],
    outputs=gr.Textbox(label="Сгенерированный текст"),
    title="Генерация текста с GPT",
    description="Введите промпт и настройте параметры для генерации текста с помощью вашей модели GPT.",
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
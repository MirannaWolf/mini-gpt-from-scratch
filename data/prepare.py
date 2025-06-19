import json
import numpy as np
from train import CharTokenizer

def prepare_data():
    # Load text
    with open('data/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Initialize tokenizer
    tokenizer = CharTokenizer(text)
    
    # Encode text
    data = np.array(tokenizer.encode(text), dtype=np.uint16)
    
    # Split into train and val
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    # Save binary files
    train_data.tofile('data/train.bin')
    val_data.tofile('data/val.bin')
    
    # Save vocabulary
    with open('data/vocab.json', 'w', encoding='utf-8') as f:
        json.dump({'stoi': tokenizer.stoi, 'itos': tokenizer.itos}, f, ensure_ascii=False)
    
    print("Data preparation completed. Train and val data saved to data/train.bin and data/val.bin")

if __name__ == "__main__":
    prepare_data()
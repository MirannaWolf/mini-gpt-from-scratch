class Config:
    batch_size = 64
    block_size = 128
    max_iters = 5000
    learning_rate = 3e-4
    n_embd = 256
    n_head = 8
    n_layer = 6
    dropout = 0.1
    eval_interval = 500
    save_interval = 1000
    vocab_size = 65  # Placeholder, updated in train.py
    device = 'cpu'   # Updated in train.py
config = Config()
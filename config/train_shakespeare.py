class Config:
    batch_size = 16        
    block_size = 128      
    max_iters = 5000       
    learning_rate = 3e-4   
    n_embd = 128           
    n_head = 8             
    n_layer = 4            
    dropout = 0.1          
    eval_interval = 500    
    save_interval = 1000   
    vocab_size = 65       
    device = 'cpu'        

config = Config()
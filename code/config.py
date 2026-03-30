class GPTConfig:
    def __init__(self, **kwargs):
        self.vocab_size = 256
        self.ctx_len = kwargs.get('ctx_len', 1024) #From git repo
        #46M parameter model (from paper)
        self.n_embd = kwargs.get('n_embd', 512)
        self.n_ffn = kwargs.get('n_ffn', 4 * kwargs.get('n_embd', 512)) 
        self.n_layer = kwargs.get('n_layer', 12)
        
        #216M parameter model
        #self.n_embd = kwargs.get('n_embd', 768)
        #self.n_layer = kwargs.get('n_layer', 24)

        self.lif_beta = kwargs.get('lif_beta', 0.5) #From paper
        self.lif_threshold = kwargs.get('lif_threshold', 1.0) #From paper
        self.lif_alpha = kwargs.get('lif_alpha', 2.0) #From git repo
        

class TrainerConfig:
    def __init__(self, **kwargs):
        # Learning rate schedule (from git repo train.py)
        self.learning_rate = kwargs.get('learning_rate', 6e-4)
        self.lr_final = kwargs.get('lr_final', 1e-5)
        self.lr_decay = kwargs.get('lr_decay', True)
        self.warmup_tokens = kwargs.get('warmup_tokens', 0) 

        # Optimizer (from git repo train.py)
        self.betas = kwargs.get('betas', (0.9, 0.99))
        self.eps = kwargs.get('eps', 4e-9)
        self.grad_norm_clip = kwargs.get('grad_norm_clip', 1.0)

        # Training loop (from git repo train.py)
        self.max_epochs = kwargs.get('max_epochs', 1000)
        self.batch_size = kwargs.get('batch_size', 12)
        self.epoch_length_fixed = kwargs.get('epoch_length_fixed', 10000)
        self.num_workers = kwargs.get('num_workers', 0)

        # Checkpointing (from git repo train.py)
        self.epoch_save_frequency = kwargs.get('epoch_save_frequency', 10)
        self.epoch_save_path = kwargs.get('epoch_save_path', 'results/checkpoints')

        # Logging 
        self.log_every = kwargs.get('log_every', 50)
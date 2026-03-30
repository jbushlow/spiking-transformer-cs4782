from pathlib import Path
import torch
from torch.utils.data import DataLoader
from dataset import Enwik8Dataset
from config import GPTConfig, TrainerConfig

model_config = GPTConfig()
trainer_config = TrainerConfig()

data_dir = Path(__file__).resolve().parent.parent / "data" / "enwik8_split"

train_dataset = Enwik8Dataset(data_dir / "train.txt", model_config.ctx_len)
val_dataset   = Enwik8Dataset(data_dir / "valid.txt", model_config.ctx_len)
test_dataset  = Enwik8Dataset(data_dir / "test.txt", model_config.ctx_len)

train_loader = DataLoader(
    train_dataset,
    batch_size=trainer_config.batch_size,
    shuffle=True,
    num_workers=trainer_config.num_workers,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=trainer_config.batch_size,
    shuffle=False,
    num_workers=trainer_config.num_workers,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=trainer_config.batch_size,
    shuffle=False,
    num_workers=trainer_config.num_workers,
)

import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from config import get_sanity_model_config, TrainerConfig
from nlu.nlu_model import SpikingGPTClassifier
from nlu.nlu_dataset import SST2Dataset
from nlu.train_cls import train

CHECKPOINT = Path(__file__).resolve().parent.parent.parent / 'results' / 'checkpoints' / 'epoch_4_17.pt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

gpt_config = get_sanity_model_config()
model = SpikingGPTClassifier(gpt_config, num_classes=2).to(DEVICE)

ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
model.backbone.load_state_dict(ckpt['model_state_dict'], strict=False)
print(f"Loaded backbone from {CHECKPOINT}")

train_dataset = SST2Dataset(split='train')
val_dataset = SST2Dataset(split='validation')

trainer_config = TrainerConfig(
    max_epochs=5,
    batch_size=32,
    learning_rate=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
)

train(model, train_dataset, val_dataset, trainer_config, DEVICE, ctx_len=gpt_config.ctx_len)

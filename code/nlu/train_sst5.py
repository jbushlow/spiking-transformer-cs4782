import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from config import get_spikegpt_46m_config, TrainerConfig
from nlu.nlu_model import SpikingGPTClassifier
from nlu.nlu_dataset import SST5Dataset
from nlu.train_cls import train

CHECKPOINT = 'results/checkpoints/epoch_1.pt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

gpt_config = get_spikegpt_46m_config()
model = SpikingGPTClassifier(gpt_config, num_classes=5).to(DEVICE)

ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
model.backbone.load_state_dict(ckpt['model_state_dict'], strict=False)
print(f"Loaded backbone from {CHECKPOINT}")

train_dataset = SST5Dataset(split='train')
val_dataset = SST5Dataset(split='validation')

trainer_config = TrainerConfig(
    max_epochs=5,
    batch_size=32,
    learning_rate=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
)

train(model, train_dataset, val_dataset, trainer_config, DEVICE, ctx_len=gpt_config.ctx_len)

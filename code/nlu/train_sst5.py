import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch

from config import TrainerConfig, get_spikegpt_46m_config
from nlu.nlu_model import SpikingGPTClassifier
from nlu.nlu_dataset import SST5Dataset
from nlu.train_cls import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="LM checkpoint to initialize the 46M backbone")
    parser.add_argument("--output_dir", default="results/nlu_checkpoints/sst5_46m")
    parser.add_argument("--ctx_len", type=int, default=1024)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpt_config = get_spikegpt_46m_config(ctx_len=args.ctx_len)
    model = SpikingGPTClassifier(gpt_config, num_classes=5).to(device)

    ckpt = torch.load(Path(args.checkpoint), map_location=device)
    backbone_state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.backbone.load_state_dict(backbone_state, strict=False)
    print(f"Loaded backbone from {args.checkpoint}")

    train_dataset = SST5Dataset(split='train')
    val_dataset = SST5Dataset(split='validation')

    trainer_config = TrainerConfig(
        max_epochs=5,
        batch_size=32,
        learning_rate=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    trainer_config.cls_checkpoint_dir = args.output_dir

    train(model, train_dataset, val_dataset, trainer_config, device, ctx_len=gpt_config.ctx_len)


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
from torch.nn import functional as F

from model import SpikingGPT


class SpikingGPTClassifier(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.backbone = SpikingGPT(config)
        self.classifier = nn.Linear(config.n_embd, num_classes)

    def encode(self, idx):
        x = self.backbone.emb(idx)
        for block in self.backbone.blocks:
            x = block(x)
        x = self.backbone.ln_out(x)
        return x


    def forward(self, idx, labels=None):
        x = self.encode(idx)
        pooled = x.mean(dim=1)
        logits = self.classifier(pooled)

        if labels is not None:
            return F.cross_entropy(logits, labels)

        return logits


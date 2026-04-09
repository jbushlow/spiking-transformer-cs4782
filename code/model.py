import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from spikingjelly.activation_based import neuron, functional, surrogate
from spikingjelly.activation_based.surrogate import ATan as atan


class BinaryEmbedding(nn.Module):
    def __init__(self, vocab_size, n_embd):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, n_embd)
        self.atan = atan()

    def forward(self, x):
        #atan in the forward pass converts continous embedding values to binary 0s and 1s
        #in the backward pass, the surrogate gradient allows for non-zero gradients to flow through the binary activations
        return self.atan(self.emb(x))
    
#Next step:
#Implement RWKV_TimeMix which is the equivalent of a transformer attention block
#Cannot implement with CUDA like the paper
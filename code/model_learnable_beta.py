import torch
import torch.nn as nn
from spikingjelly.activation_based import surrogate

from model import BinaryEmbedding
from model import SpikeBlock as BaseSpikeBlock
from model import SpikingGPT as BaseSpikingGPT
from model import SpikingRFFN as BaseSpikingRFFN
from model import SpikingRWKV as BaseSpikingRWKV


class LearnableLIF(nn.Module):
    def __init__(self, n_embd, init_beta=0.5, threshold=1.0, alpha=2.0):
        super().__init__()
        init_beta = min(max(init_beta, 1e-4), 1 - 1e-4)
        self.beta_raw = nn.Parameter(torch.full((n_embd,), torch.logit(torch.tensor(init_beta))))
        self.threshold = threshold
        self.spike_fn = surrogate.ATan(alpha=alpha)

    def forward(self, x):
        # x is [T, B, C]
        T, B, C = x.shape
        beta = torch.sigmoid(self.beta_raw).to(device=x.device, dtype=x.dtype).view(1, C)
        mem = torch.zeros(B, C, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(T):
            mem = beta * mem + x[t]
            spike = self.spike_fn(mem - self.threshold)
            mem = mem * (1 - spike)
            outputs.append(spike)

        return torch.stack(outputs, dim=0)


class SpikingRWKVLearnableBeta(BaseSpikingRWKV):
    def __init__(self, config, layer_id):
        super().__init__(config, layer_id)
        self.spike = LearnableLIF(
            self.n_embd,
            init_beta=config.lif_beta,
            threshold=config.lif_threshold,
            alpha=config.lif_alpha,
        )


class SpikingRFFNLearnableBeta(BaseSpikingRFFN):
    def __init__(self, config, layer_id):
        super().__init__(config, layer_id)
        self.spike = LearnableLIF(
            self.n_embd,
            init_beta=config.lif_beta,
            threshold=config.lif_threshold,
            alpha=config.lif_alpha,
        )


class SpikeBlockLearnableBeta(BaseSpikeBlock):
    def __init__(self, config, layer_id):
        nn.Module.__init__(self)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.layer_id = layer_id
        if layer_id == 0:
            self.ln0 = nn.LayerNorm(config.n_embd)
        self.dropout = nn.Dropout(0.03)

        self.spiking_rwkv = SpikingRWKVLearnableBeta(config, layer_id)
        self.spiking_rffn = SpikingRFFNLearnableBeta(config, layer_id)


class SpikingGPTLearnableBeta(BaseSpikingGPT):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.ctx_len = config.ctx_len
        self.n_embd = config.n_embd
        self.vocab_size = config.vocab_size

        self.emb = BinaryEmbedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList(
            [SpikeBlockLearnableBeta(config, layer_id=i) for i in range(config.n_layer)]
        )
        self.ln_out = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

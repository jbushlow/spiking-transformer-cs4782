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

class SpikingRWKV(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.n_embd = config.n_embd
        self.layer_id = layer_id
        self.ctx_len = config.ctx_len

        self._init_rwkv_params(config,layer_id)
        self._init_layers(config)
        
    def _init_rwkv_params(self,config,layer_id):

        #So these ratios are a way to make different layers initialize differently.

        #That matters because SpikeGPT does not want every layer to start identically. 
        # Early layers and late layers are supposed to behave a bit differently.

        # increases as layers go deper
        ratio_0_to_1 = layer_id/max(config.n_layer-1,1)
        # decreases as layers go deeper
        ratio_1_to_almost0 = 1 - (layer_id/config.n_layer)

        # creates one learnable decay value per embedding channel
        # some features will have short term memory or long term memory
        decay_speed = torch.ones(self.n_embd)
        for h in range(self.n_embd):
            decay_speed[h] = -5 + 8 * (h / max(self.n_embd - 1, 1)) ** (0.7 + 1.3 * ratio_0_to_1)
        self.time_decay = nn.Parameter(decay_speed)

        # time first controls how the current token enters recurrence
        # affects how current tiemstep is weighed relative to runnign history of recurrence in rwkv
        # this just slighlty offsets neighboring channels so they don't start identical
        zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(self.n_embd)], dtype=torch.float32) * 0.5
        self.time_first = nn.Parameter(torch.ones(self.n_embd) * math.log(0.3) + zigzag)

        # mixing coefficients that tell model how much to use the current time step input
        # versus the pervious timestep input. 
        x = torch.ones(1, 1, self.n_embd)
        for i in range(self.n_embd):
            x[0, 0, i] = i / self.n_embd
        self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        self.time_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
        self.time_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))
        

    def _init_layers(self,config):
        self.time_shift = nn.ZeroPad2d((0,0,1,-1))
        self.key = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.value = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.receptance = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.output = nn.Linear(self.n_embd, self.n_embd, bias=False)
       

        self.spike = neuron.MultiStepLIFNode(
            tau = 2,
            surrogate_function = surrogate.ATan(alpha = config.lif_alpha),
            v_threshold = config.lif_threshold,
            backend = "torch",
        )

    def _time_mix_inputs(self,x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        return xk, xv, xr


    def _compute_kvr(self,xk,xv,xr):
        k = self.key(xk)
        v = self.value(xv)
        r = torch.sigmoid(self.receptance(xr))
        return k,v,r

    def _wkv_recurrence(self,k,v):
        B, T, C = k.shape
        # per channel decay from time decay
        w = -torch.exp(self.time_decay).view(1, C)

        # per channel bias for current time step from time first
        u = self.time_first.view(1, C)

        # running weighted sum of past values
        aa = torch.zeros(B, C, device=k.device, dtype=k.dtype)

        # running weighted sum of past weights
        bb = torch.zeros(B, C, device=k.device, dtype=k.dtype)

        # running normalization helper for stability
        pp = torch.full((B, C), -1e30, device=k.device, dtype=k.dtype)

        # stores output at each timestep
        outputs = []

        for t in range(T):
            kk = k[:, t, :]
            vv = v[:, t, :]

            ww = u + kk
            p = torch.maximum(pp, ww)
            e1 = torch.exp(pp - p)
            e2 = torch.exp(ww - p)

            a = e1 * aa + e2 * vv
            b = e1 * bb + e2
            y = a / (b + 1e-9)
            outputs.append(y)

            ww = pp + w
            p = torch.maximum(ww, kk)
            e1 = torch.exp(ww - p)
            e2 = torch.exp(kk - p)

            aa = e1 * aa + e2 * vv
            bb = e1 * bb + e2
            pp = p

        return torch.stack(outputs, dim=1)
    
    def _apply_spike(self,x):
        x = x.permute(1,0,2)
        x = self.spike(x)
        return x.permute(1,0,2)

    def forward(self,x):
        xk, xv, xr = self._time_mix_inputs(x)
        k, v, r = self._compute_kvr(xk, xv, xr)
        rwkv = self._wkv_recurrence(k, v)
        out = self.output(r * rwkv)
        return self._apply_spike(out)

class SpikingRFFN(nn.Module):
   
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.ctx_len = config.ctx_len
        self.n_embd = config.n_embd

        self._init_rffn_params(config,layer_id)
        self._init_layers(config)


    def _init_layers(self,config):
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        hidden_sz = 4 * self.n_embd
        self.key = nn.Linear(self.n_embd, hidden_sz, bias=False)
        self.receptance = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, self.n_embd, bias=False)

        self.spike = neuron.MultiStepLIFNode(
            tau=2.0,
            surrogate_function=surrogate.ATan(alpha=config.lif_alpha),
            v_threshold=config.lif_threshold,
            backend="torch",
        )
    
    
    def _init_rffn_params(self,config,layer_id):
        ratio_1_to_almost0 = 1 - (layer_id/config.n_layer)

        x = torch.ones(1,1,self.n_embd)
        for i in range(self.n_embd):
            x[0, 0, i] = i / self.n_embd

        self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        
    def _time_mix_inputs(self,x):
        # gives model access to the token from one step earlier
        xx = self.time_shift(x)

        # weighted mixure of current input x and previous input xx
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx *(1- self.time_mix_r)
        return xk,xr

    def _compute_rffn(self,xk,xr):
        k = self.key(xk)
        k = torch.square(torch.relu(k))
        kv = self.value(k)
        r = torch.sigmoid(self.receptance(xr))
        return r*kv

    def _apply_spike(self,x):
        x = x.permute(1,0,2)
        x = self.spike(x)
        return x.permute(1,0,2)


    def forward(self,x):
        xk,xr = self._time_mix_inputs(x)
        out =  self._compute_rffn(xk,xr)
        out = self.spike(out.permute(1,0,2)).permute(1,0,2)
        return out

class SpikeBlock(nn.Module):
    def __init__(self,config,layer_id):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        self.spiking_rwkv = SpikingRWKV(config, layer_id)
        self.spiking_rffn = SpikingRFFN(config, layer_id)

    def forward(self,x):
        x = x + self.spiking_rwkv(self.ln1(x))
        x = x + self.spiking_rffn(self.ln2(x))
        return x

class SpikingGPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ctx_len = config.ctx_len
        self.n_embd = config.n_embd
        self.vocab_size = config.vocab_size

        self.emb = BinaryEmbedding(config.vocab_size, config.n_embd)

        self.blocks = nn.ModuleList(
            [SpikeBlock(config, layer_id=i) for i in range(config.n_layer)]
        )

        self.ln_out = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    
    def forward(self,idx, targets = None):
        B,T = idx.shape
        assert T <= self.ctx_len, "Input sequence is longer than ctx_len."
        x = self.emb(idx)
        for block in self.blocks:
            x = block(x)

        x = self.ln_out(x)
        logits = self.head(x)

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
            return loss
        return logits


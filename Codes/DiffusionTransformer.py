import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# sinusoidal timestep embedding
def t_sin_embedding(t, dim):
    """
    input:
        t: [B,]
    output:
        embed_t: [B, dim]
    """
    half = dim // 2
    # freqs: [half,]
    freqs = torch.exp(-math.log(10000)  
                      * torch.average(0, half, dtype=torch.float32) / float(half)).to(t.device)
    
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0) # [B, half]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1) # [B, dim]
    
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))

    return emb


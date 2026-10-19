"""Original candidate-conditioned extension; frozen Stage 4 encoder/trajectory head."""
import torch
from torch import nn


class ActionConditionedWorld(nn.Module):
    def __init__(self, base, conditioned=True):
        super().__init__()
        self.base=base
        self.conditioned=conditioned
        d=base.config.predictor_dim
        self.command_embedding=nn.Sequential(nn.Linear(40,d),nn.GELU(),nn.Linear(d,d))
        nn.init.zeros_(self.command_embedding[-1].weight)
        nn.init.zeros_(self.command_embedding[-1].bias)
        self.register_buffer('command_scale',torch.tensor([20.,5.,10.,1.]))
        for p in self.base.parameters():p.requires_grad_(False)
        for module in (self.base.history_fusion,self.base.predictor,self.command_embedding):
            for p in module.parameters():p.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        self.base.encoder.eval()
        return self

    def forward(self, history, ego, command):
        if history.ndim!=4 or history.shape[1]!=4 or ego.shape[1:]!=(4,2):
            raise ValueError('Expected encoded history [B,4,N,D] and observed motion [B,4,2]')
        if command.shape!=(len(history),10,4):raise ValueError('Expected commanded, not realized, [B,10,4] trajectory')
        if not torch.isfinite(command).all():raise ValueError('Nonfinite command')
        b,_,n,d=history.shape;current=history[:,-1]
        differences=(history[:,:3]-current[:,None]).permute(0,2,1,3).reshape(b,n,3*d)
        motion=ego.flatten(1)
        fused=current+self.base.history_fusion(torch.cat([differences,motion[:,None].expand(-1,n,-1)],-1))
        # Unconditional control retains exactly the same parameters and initialization.
        action=(command/self.command_scale).flatten(1)
        if not self.conditioned:action=torch.zeros_like(action)
        p=self.base.predictor
        hidden=p.input(fused)+p.ego(ego[:,-1])[:,None]+p.position+self.command_embedding(action)[:,None]
        delta=p.output(p.blocks(hidden)).reshape(b,n,4,d).permute(0,2,1,3)
        return current[:,None]+delta


def masked_world_error(prediction, target, valid):
    error=(prediction.float()-target.float()).square().mean((-1,-2))
    return (error*valid).sum()/valid.sum().clamp_min(1)

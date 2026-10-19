"""Spatially preserved four-history/four-future deterministic world/action model."""
import torch
from torch import nn

from models.latent import BEVEncoder, OneStepPredictor


def constant_turn_trajectory(ego, steps=10, dt=.2):
    """Parameter-free constant-speed/yaw-rate prior from observed current motion."""
    t = torch.arange(1, steps+1, device=ego.device, dtype=ego.dtype)*dt
    speed, yaw = ego[:, -1, 0:1]*10, ego[:, -1, 1:2]
    angle = yaw*t
    x = speed*t*torch.sinc(angle/torch.pi)
    y = speed*t*(angle/2)*torch.sinc(angle/(2*torch.pi)).square()
    return torch.stack([x, y, speed.expand_as(x), angle], dim=-1)


class JointModel(nn.Module):
    def __init__(self, config, normalizer):
        super().__init__(); self.config = config
        self.encoder = BEVEncoder(config)
        self.predictor = OneStepPredictor(config, self.encoder.token_count)
        self.predictor.output = nn.Linear(config.predictor_dim, config.future_frames*config.encoder_dim)
        self.history_fusion = nn.Linear(3*config.encoder_dim+8, config.encoder_dim)
        nn.init.zeros_(self.history_fusion.weight); nn.init.zeros_(self.history_fusion.bias)
        self.action_projection = nn.Sequential(nn.Linear(5*config.encoder_dim, config.action_projection), nn.GELU())
        self.action_head = nn.Sequential(
            nn.Linear(self.encoder.token_count*config.action_projection+8, config.action_hidden), nn.GELU(),
            nn.Linear(config.action_hidden, config.action_hidden), nn.GELU(),
            nn.Linear(config.action_hidden, 40))
        nn.init.zeros_(self.action_head[-1].weight); nn.init.zeros_(self.action_head[-1].bias)
        self.register_buffer('trajectory_mean', torch.tensor(normalizer.trajectory_mean))
        self.register_buffer('trajectory_std', torch.tensor(normalizer.trajectory_std))
        self.register_buffer('trajectory_scale', torch.tensor(float(normalizer.scale)))

    def initialize_stage3(self, state):
        self.encoder.load_state_dict({key[8:]:value for key,value in state['model'].items() if key.startswith('encoder.')})
        shared = {key[10:]:value for key,value in state['model'].items()
                  if key.startswith('predictor.') and not key.startswith('predictor.output.')}
        missing, unexpected = self.predictor.load_state_dict(shared, strict=False)
        if set(missing) != {'output.weight', 'output.bias'} or unexpected:
            raise ValueError((missing, unexpected))
        with torch.no_grad():
            for horizon in range(4):
                start = horizon*self.config.encoder_dim; stop = start+self.config.encoder_dim
                self.predictor.output.weight[start:stop].copy_(state['model']['predictor.output.weight']*(horizon+1))
                self.predictor.output.bias[start:stop].copy_(state['model']['predictor.output.bias']*(horizon+1))

    def forward_encoded(self, history, ego):
        if history.ndim != 4 or history.shape[1] != 4 or ego.shape[1:] != (4, 2):
            raise ValueError('Expected [B,4,N,D] encoded history and [B,4,2] observed motion')
        b, _, n, d = history.shape
        current = history[:, -1]
        differences = (history[:, :3]-current[:, None]).permute(0, 2, 1, 3).reshape(b, n, 3*d)
        motion = ego.flatten(1)
        fused = current + self.history_fusion(torch.cat([differences, motion[:, None].expand(-1,n,-1)],-1))
        hidden = self.predictor.input(fused)+self.predictor.ego(ego[:, -1])[:, None]+self.predictor.position
        hidden = self.predictor.blocks(hidden)
        delta = self.predictor.output(hidden).reshape(b, n, 4, d).permute(0, 2, 1, 3)
        future = current[:, None]+delta
        # Preserve every spatial location; never average tokens into a global vector.
        content = torch.cat([current, future.permute(0, 2, 1, 3).reshape(b,n,4*d)],-1)
        spatial = self.action_projection(content).flatten(1)
        residual = self.action_head(torch.cat([spatial, motion],-1)).reshape(b, 10, 4)
        raw_prior = constant_turn_trajectory(ego.float())
        scale = torch.stack([self.trajectory_scale]*3+[self.trajectory_scale.new_tensor(1.)])
        prior = (raw_prior*scale-self.trajectory_mean)/self.trajectory_std
        return dict(trajectory=prior+residual.float(), latent=future, current=current)

    def forward(self, history, ego, future_bev=None):
        if history.ndim != 5 or history.shape[1:] != (4, 4, 128, 128):
            raise ValueError('Expected uint8 [B,4,4,128,128] BEV history')
        b = len(history)
        observed = history.flatten(0, 1)
        frames = observed if future_bev is None else torch.cat([observed, future_bev.flatten(0, 1)])
        encoded = self.encoder(frames)
        result = self.forward_encoded(encoded[:b*4].reshape(b,4,self.encoder.token_count,-1), ego)
        if future_bev is not None:
            result['target'] = encoded[b*4:].reshape(b,4,self.encoder.token_count,-1)
        return result


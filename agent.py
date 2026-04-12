from torch import nn


def layer_init(layer, gain: float = 2**0.5, bias: float = 0.0):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, bias)
    return layer


class Agent(nn.Module):
    def __init__(self, n_frame_stack: int, n_actions: int):
        super().__init__()
        # grayscale Atari frame size = (84, 84).
        self.net = nn.Sequential(
            layer_init(nn.Conv2d(n_frame_stack, 32, 8, stride=4, padding=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 8 * 8, 64)),
            nn.ReLU(),
        )
        self.policy = layer_init(nn.Linear(64, n_actions), gain=0.01)
        self.critic = layer_init(nn.Linear(64, 1), gain=1.0)

    def forward(self, obs):
        obs = obs / 255.0
        z = self.net(obs)
        return self.policy(z), self.critic(z).squeeze(-1)

    def get_action_logits(self, obs):
        obs = obs / 255.0
        return self.policy(self.net(obs))
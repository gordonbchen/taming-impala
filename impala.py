"""IMPALA: https://arxiv.org/pdf/1802.01561."""
from dataclasses import dataclass
import envpool
import gymnasium as gym
import numpy as np
import torch
from cli_params import CLIParams
from torch import nn, Tensor
from torch.distributions import Categorical
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

torch.set_float32_matmul_precision("high")


@dataclass
class HyperParams(CLIParams):
    n_epochs: int = 4000
    n_steps: int = 128
    n_envs: int = 16
    device: str = "cuda"

    lr: float = 3e-4
    update_steps: int = 4

    discount_gamma: float = 0.99
    impala_max_rho: float = 1.0
    impala_max_c: float = 1.0
    critic_loss_coeff: float = 0.5
    entropy_coeff: float = 0.01
    max_grad_norm: float = 0.5

    n_frame_stack: int = 4

    output_dir: str = "runs/test"

HP = HyperParams()


def layer_init(layer, gain: float = 2**0.5, bias: float = 0.0):
    torch.nn.init.orthogonal_(layer.weight, gain)
    torch.nn.init.constant_(layer.bias, bias)
    return layer

class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Conv2d(HP.n_frame_stack, 32, 8, stride=4, padding=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 8 * 8, 64)),
            nn.ReLU(),
        )
        # 0 = NOOP, 1 = LEFT, 2 = RIGHT.
        self.ACTION_MAP = torch.tensor([0, 3, 2], dtype=torch.int64)
        self.policy = layer_init(nn.Linear(64, len(self.ACTION_MAP)), gain=0.01)
        self.critic = layer_init(nn.Linear(64, 1), gain=1.0)

    def forward(self, obs):
        obs = obs / 255.0
        z = self.net(obs)
        return self.policy(z), self.critic(z).squeeze(-1)

    def get_action_logits(self, obs):
        obs = obs / 255.0
        return self.policy(self.net(obs))


class EnvPoolEpisodeStats(gym.Wrapper):
    def __init__(self, env, n_envs: int):
        super().__init__(env)
        self.episode_returns = np.zeros(n_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(n_envs, dtype=np.int32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode_returns[:] = 0.0
        self.episode_lengths[:] = 0
        return obs, info

    def step(self, actions):
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        dones = np.logical_or(terminated, truncated)

        self.episode_returns += rewards
        self.episode_lengths += 1

        if dones.any():
            info["stats"] = {
                "done_mask": dones,
                "returns": self.episode_returns.copy(),
                "lens": self.episode_lengths.copy(),
            }
            self.episode_returns[dones] = 0.0
            self.episode_lengths[dones] = 0

        return obs, rewards, terminated, truncated, info


@torch.no_grad()
def calc_vtrace_targets(values: Tensor, rewards: Tensor, dones: Tensor, log_probs: Tensor, old_log_probs: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    next_nonterminal = 1.0 - dones[1:]
    td_deltas = rewards + next_nonterminal * HP.discount_gamma * values[1:] - values[:-1]

    action_ratios = (log_probs - old_log_probs).exp()
    rho = action_ratios.clamp(max=HP.impala_max_rho)
    c = action_ratios.clamp(max=HP.impala_max_c)

    vtrace_deltas = torch.zeros_like(values)
    for t in reversed(range(HP.n_steps)):
        vtrace_deltas[t] = rho[t] * td_deltas[t] + next_nonterminal[t] * HP.discount_gamma * c[t] * vtrace_deltas[t+1]
    vtrace_targets = vtrace_deltas + values

    advantages = rewards + next_nonterminal * HP.discount_gamma * vtrace_targets[1:] - values[:-1]

    return vtrace_targets, advantages, rho, action_ratios


# Create envs.
envs = envpool.make_gymnasium("Pong-v5", num_envs=HP.n_envs)
envs = EnvPoolEpisodeStats(envs, n_envs=HP.n_envs)

# Create agent.
agent = Agent().to(device=HP.device)
agent.compile()
optim = Adam(agent.parameters(), lr=HP.lr, fused=True)

# Storage buffers.
OBS_SHAPE = (HP.n_frame_stack, 84, 84)
obss = torch.zeros((HP.n_steps+1, HP.n_envs) + OBS_SHAPE, dtype=torch.float32, device=HP.device)
dones = torch.zeros((HP.n_steps+1, HP.n_envs), dtype=torch.float32, device=HP.device)
actions = torch.zeros((HP.n_steps, HP.n_envs), dtype=torch.int64, device=HP.device)
rewards = torch.zeros((HP.n_steps, HP.n_envs), dtype=torch.float32, device=HP.device)
old_log_probs = torch.zeros((HP.n_steps, HP.n_envs), dtype=torch.float32, device=HP.device)

obs, _ = envs.reset()
obs = torch.tensor(obs, dtype=torch.float32, device=HP.device)
done = torch.zeros(HP.n_envs, dtype=torch.float32, device=HP.device)

log = SummaryWriter(log_dir=HP.output_dir)
for epoch in range(HP.n_epochs):
    total_reward = 0.0
    episode_length = 0
    n_episodes = 0

    # obs and done continue from prev epoch
    obss[0] = obs
    dones[0] = done
    for t in range(HP.n_steps):
        # import matplotlib.pyplot as plt
        # plt.imshow(obs[0][0].cpu().numpy(), cmap="gray")
        # plt.show()
        with torch.no_grad():
            logits = agent.get_action_logits(obs)
        action_dist = Categorical(logits=logits)
        action = action_dist.sample()

        actions[t] = action
        old_log_probs[t] = action_dist.log_prob(action)

        obs, reward, terminated, truncated, info = envs.step(agent.ACTION_MAP[action.cpu()].numpy())
        rewards[t] = torch.tensor(reward, dtype=torch.float32, device=HP.device)

        obs = torch.tensor(obs, dtype=torch.float32, device=HP.device)
        done = torch.tensor(terminated | truncated, dtype=torch.float32, device=HP.device)
        obss[t+1] = obs
        dones[t+1] = done

        if "stats" in info:
            done_mask = info["stats"]["done_mask"]
            total_reward += info["stats"]["returns"][done_mask].sum()
            episode_length += info["stats"]["lens"][done_mask].sum()
            n_episodes += int(done_mask.sum())

    # Linearly decay lr to 0.
    optim.param_groups[0]["lr"] = HP.lr - (HP.lr / HP.n_epochs) * epoch

    for _ in range(HP.update_steps):
        logits, values = agent(obss.view(-1, *OBS_SHAPE))
        logits, values = [t.view(-1, HP.n_envs, *t.shape[1:]) for t in (logits, values)]
        action_dist = Categorical(logits=logits[:-1])
        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy().mean()

        vtrace_targets, advantages, rho, action_ratios = calc_vtrace_targets(values, rewards, dones, log_probs, old_log_probs)
        critic_loss = 0.5 * (vtrace_targets[:-1] - values[:-1]).square().mean()
        policy_loss = -(rho * advantages * log_probs).mean()

        loss = policy_loss + (HP.critic_loss_coeff * critic_loss) - (HP.entropy_coeff * entropy)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=HP.max_grad_norm).item()
        optim.step()

    # Log.
    mean_total_reward = None
    if n_episodes > 0:
        mean_total_reward = total_reward / n_episodes
        log.add_scalar("ep_stats/reward", mean_total_reward, epoch)
        log.add_scalar("ep_stats/length", episode_length / n_episodes, epoch)
    log.add_scalar("ep_stats/episodes", n_episodes, epoch)
    log.add_scalar("loss/policy", policy_loss.item(), epoch)
    log.add_scalar("loss/critic", critic_loss.item(), epoch)
    log.add_scalar("loss/entropy", entropy.item(), epoch)
    log.add_scalar("train/grad_norm", grad_norm, epoch)
    log.add_scalar("train/lr", optim.param_groups[0]["lr"], epoch)
    log.add_scalar("train/action_ratios", action_ratios.mean().item(), epoch)
    print(f"{epoch}: reward={mean_total_reward}")

log.close()
envs.close()

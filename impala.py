"""PPO, heavily inspired by PufferLib and CleanRL."""
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import numpy as np
from cli_params import CLIParams
import envpool

torch.set_float32_matmul_precision("high")


@dataclass
class HyperParams(CLIParams):
    n_epochs: int = 5000
    n_steps: int = 128
    n_envs: int = 16
    device: str = "cuda"

    lr: float = 3e-4
    ppo_epochs: int = 4
    minibatch_size: int = 256

    discount_gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    critic_loss_coeff: float = 0.5
    entropy_coeff: float = 0.01
    max_grad_norm: float = 0.5

    n_frame_stack: int = 4

    output_dir: str = "runs/test"

    def check_args(self):
        assert (self.n_envs * self.n_steps) % self.minibatch_size == 0

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
            nn.ReLU()
        )
        # 0 = NOOP, 1 = LEFT, 2 = RIGHT.
        self.ACTION_MAP = torch.tensor([0, 3, 2], dtype=torch.int64)
        self.policy = layer_init(nn.Linear(64, len(self.ACTION_MAP)), gain=0.01)
        self.critic = layer_init(nn.Linear(64, 1), gain=1)

    def forward(self, obs):
        obs = obs / 255.0
        z = self.net(obs)
        return self.policy(z), self.critic(z).squeeze(-1)
    
    def get_values(self, obs):
        obs = obs / 255.0
        return self.critic(self.net(obs)).squeeze(-1)
    
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
        self.episode_returns[:] = 0
        self.episode_lengths[:] = 0
        return obs, info

    def step(self, actions):
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        dones = np.logical_or(terminated, truncated)

        self.episode_returns += rewards
        self.episode_lengths += 1

        if dones.any():
            info["stats"] = {}
            info["stats"]["done_mask"] = dones

            # Only meaningful where dones == True
            info["stats"]["returns"] = self.episode_returns.copy()
            info["stats"]["lens"] = self.episode_lengths.copy()

            self.episode_returns[dones] = 0.0
            self.episode_lengths[dones] = 0

        return obs, rewards, terminated, truncated, info

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
advantages = torch.zeros((HP.n_steps, HP.n_envs), dtype=torch.float32, device=HP.device)
old_log_probs = torch.zeros((HP.n_steps, HP.n_envs), dtype=torch.float32, device=HP.device)
values = torch.zeros((HP.n_steps+1, HP.n_envs), dtype=torch.float32, device=HP.device)

obs, _ = envs.reset()
obs = torch.tensor(obs, dtype=torch.float32, device=HP.device)
done = torch.zeros(HP.n_envs, dtype=torch.float32, device=HP.device)

log = SummaryWriter(log_dir=HP.output_dir)
for epoch in range(HP.n_epochs):
    total_reward = 0
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
            logits, value = agent(obs)
        action_dist = Categorical(logits=logits)
        action = action_dist.sample()

        actions[t] = action
        old_log_probs[t] = action_dist.log_prob(action)
        values[t] = value

        obs, reward, terminated, truncated, info = envs.step(agent.ACTION_MAP[action.cpu()].numpy())
        rewards[t] = torch.tensor(reward, dtype=torch.float32, device=HP.device)

        obs = torch.tensor(obs, dtype=torch.float32, device=HP.device)
        done = torch.tensor(terminated | truncated, dtype=torch.float32, device=HP.device)
        obss[t+1] = obs
        dones[t+1] = done

        if "stats" in info.keys():
            done_mask = info["stats"]["done_mask"]
            total_reward += info["stats"]["returns"][done_mask].sum()
            episode_length += info["stats"]["lens"][done_mask].sum()
            n_episodes += done_mask.sum()

    # GAE.
    with torch.no_grad():
        values[-1] = agent.get_values(obs)
        gae_accum = torch.zeros(HP.n_envs, dtype=torch.float32, device=HP.device)
        for t in reversed(range(HP.n_steps)):
            # If not done, bootstrap w/ next value.
            next_nonterminal = 1 - dones[t+1]
            delta = rewards[t] + HP.discount_gamma * values[t+1] * next_nonterminal - values[t]
            advantages[t] = gae_accum = delta + HP.discount_gamma * HP.gae_lambda * gae_accum * next_nonterminal
        value_targets = advantages + values[:-1]

    # Batch buffers.
    b_obss = obss[:-1].view(-1, *OBS_SHAPE)
    b_actions, b_advantages, b_old_log_probs, b_value_targets = (
        x.view(-1) for x in (actions, advantages, old_log_probs, value_targets)
    )
    total_samples = len(b_obss)

    # Linearly decay lr to 0.
    optim.param_groups[0]["lr"] = HP.lr - (HP.lr / HP.n_epochs) * epoch

    # Optimize.
    for _ in range(HP.ppo_epochs):
        idxs = torch.randperm(total_samples, device=HP.device)
        for i in range(0, total_samples, HP.minibatch_size):
            batch_idxs = idxs[i : i+HP.minibatch_size]
            logits, value = agent(b_obss[batch_idxs])
            action_dist = Categorical(logits=logits)
            log_probs = action_dist.log_prob(b_actions[batch_idxs])
            entropy = action_dist.entropy().mean()

            # TODO: log kl + clipfrac.
            ratios = (log_probs - b_old_log_probs[batch_idxs]).exp()
            adv_mb = b_advantages[batch_idxs]
            normed_advs = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)
            pg1 = ratios * normed_advs
            pg2 = ratios.clip(1-HP.clip_eps, 1+HP.clip_eps) * normed_advs
            policy_loss = -torch.minimum(pg1, pg2).mean()

            # TODO: value clipping.
            critic_loss = 0.5 * F.mse_loss(value, b_value_targets[batch_idxs])

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
    log.add_scalar("train/grad_norm", grad_norm, global_step=epoch)
    log.add_scalar("train/lr", optim.param_groups[0]["lr"], epoch)
    print(f"{epoch}: reward={mean_total_reward}")

log.close()
envs.close()
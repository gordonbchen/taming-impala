"""IMPALA: https://arxiv.org/pdf/1802.01561."""
from dataclasses import dataclass
import envpool
import torch
from cli_params import CLIParams
from torch import Tensor
from torch.distributions import Categorical
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from agent import Agent
from env import EnvPoolEpisodeStats

torch.set_float32_matmul_precision("high")


@dataclass
class HyperParams(CLIParams):
    train_steps: int = 5_000_000
    n_rollout_steps: int = 128
    update_steps: int = 4

    lr: float = 1e-3
    adam_beta1: float = 0.1
    adam_beta2: float = 0.5

    discount_gamma: float = 0.99
    impala_max_rho: float = 1.0
    impala_max_c: float = 1.0
    critic_loss_coeff: float = 0.5
    entropy_coeff: float = 0.01
    max_grad_norm: float = 0.5

    n_envs: int = 16
    n_frame_stack: int = 4

    output_dir: str = "runs/test"
    device: str = "cuda"


@torch.no_grad()
def sample_trajectories(
        agent: Agent, envs, n_steps: int, obs: Tensor, done: Tensor,
        obss: Tensor, dones: Tensor, actions: Tensor, rewards: Tensor, old_log_probs: Tensor,
    ) -> tuple[Tensor, Tensor, float, int, int]:
    total_reward = 0.0
    total_episode_length = 0
    n_episodes = 0

    # obs and done continue from prev samples.
    obss[0] = obs
    dones[0] = done

    for t in range(n_steps):
        # import matplotlib.pyplot as plt
        # plt.imshow(obs[0][0].cpu().numpy(), cmap="gray")
        # plt.show()
        logits = agent.get_action_logits(obs)
        action_dist = Categorical(logits=logits)
        action = action_dist.sample()

        actions[t] = action
        old_log_probs[t] = action_dist.log_prob(action)

        obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
        rewards[t] = torch.tensor(reward, dtype=torch.float32, device=rewards.device)

        obs = torch.tensor(obs, dtype=torch.float32, device=obss.device)
        done = torch.tensor(terminated | truncated, dtype=torch.float32, device=dones.device)
        obss[t+1] = obs
        dones[t+1] = done

        if "stats" in info:
            done_mask = info["stats"]["done_mask"]
            total_reward += info["stats"]["returns"][done_mask].sum()
            total_episode_length += info["stats"]["lens"][done_mask].sum()
            n_episodes += int(done_mask.sum())

    return obs, done, total_reward, total_episode_length, n_episodes


def optimize_model(
        agent: Agent, obss: Tensor, actions: Tensor, rewards: Tensor, dones: Tensor, old_log_probs: Tensor, HP: HyperParams
    ) -> tuple[float, float, float, float, float]:
    logits, values = agent(obss.view(-1, *OBS_SHAPE))
    logits, values = [t.view(-1, HP.n_envs, *t.shape[1:]) for t in (logits, values)]
    action_dist = Categorical(logits=logits[:-1])
    log_probs = action_dist.log_prob(actions)
    entropy = action_dist.entropy().mean()

    vtrace_targets, advantages, rho, action_ratios = calc_vtrace_targets(values, rewards, dones, log_probs, old_log_probs, HP)
    critic_loss = 0.5 * (vtrace_targets[:-1] - values[:-1]).square().mean()
    policy_loss = -(rho * advantages * log_probs).mean()

    loss = policy_loss + (HP.critic_loss_coeff * critic_loss) - (HP.entropy_coeff * entropy)
    optim.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=HP.max_grad_norm)
    optim.step()

    return policy_loss.item(), critic_loss.item(), entropy.item(), grad_norm.item(), action_ratios.mean().item()


@torch.no_grad()
def calc_vtrace_targets(
        values: Tensor, rewards: Tensor, dones: Tensor, log_probs: Tensor, old_log_probs: Tensor, HP: HyperParams
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    next_nonterminal = 1.0 - dones[1:]
    td_deltas = rewards + next_nonterminal * HP.discount_gamma * values[1:] - values[:-1]

    action_ratios = (log_probs - old_log_probs).exp()
    rho = action_ratios.clamp(max=HP.impala_max_rho)
    c = action_ratios.clamp(max=HP.impala_max_c)

    vtrace_deltas = torch.zeros_like(values)
    for t in reversed(range(HP.n_rollout_steps)):
        vtrace_deltas[t] = rho[t] * td_deltas[t] + next_nonterminal[t] * HP.discount_gamma * c[t] * vtrace_deltas[t+1]
    vtrace_targets = vtrace_deltas + values

    advantages = rewards + next_nonterminal * HP.discount_gamma * vtrace_targets[1:] - values[:-1]

    return vtrace_targets, advantages, rho, action_ratios


if __name__ == "__main__":
    HP = HyperParams()

    # Create envs.
    envs = envpool.make_gymnasium("Pong-v5", num_envs=HP.n_envs)
    envs = EnvPoolEpisodeStats(envs, n_envs=HP.n_envs)

    # Create agent.
    agent = Agent(HP.n_frame_stack, envs.action_space.n).to(device=HP.device)
    agent.compile()
    optim = Adam(agent.parameters(), lr=HP.lr, betas=(HP.adam_beta1, HP.adam_beta2), fused=True)

    # Storage buffers.
    OBS_SHAPE = (HP.n_frame_stack, 84, 84)
    obss = torch.zeros((HP.n_rollout_steps+1, HP.n_envs) + OBS_SHAPE, dtype=torch.float32, device=HP.device)
    dones = torch.zeros((HP.n_rollout_steps+1, HP.n_envs), dtype=torch.float32, device=HP.device)
    actions = torch.zeros((HP.n_rollout_steps, HP.n_envs), dtype=torch.int64, device=HP.device)
    rewards = torch.zeros((HP.n_rollout_steps, HP.n_envs), dtype=torch.float32, device=HP.device)
    old_log_probs = torch.zeros((HP.n_rollout_steps, HP.n_envs), dtype=torch.float32, device=HP.device)

    # Env init.
    obs, _ = envs.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=HP.device)
    done = torch.zeros(HP.n_envs, dtype=torch.float32, device=HP.device)

    log = SummaryWriter(log_dir=HP.output_dir)
    global_step = 0
    while global_step < HP.train_steps:
        # Sample trajectories.
        obs, done, total_reward, total_episode_length, n_episodes = sample_trajectories(
            agent, envs, HP.n_rollout_steps, obs, done, obss, dones, actions, rewards, old_log_probs
        )
        mean_total_reward = None
        if n_episodes > 0:
            mean_total_reward = total_reward / n_episodes
            log.add_scalar("ep_stats/reward", mean_total_reward, global_step)
            log.add_scalar("ep_stats/length", total_episode_length / n_episodes, global_step)
        log.add_scalar("ep_stats/episodes", n_episodes, global_step)
        print(f"{global_step}: reward={mean_total_reward}")

        # Linearly decay lr to 0.
        optim.param_groups[0]["lr"] = HP.lr - HP.lr * (global_step / HP.train_steps)

        # Optimize.
        for _ in range(HP.update_steps):
            policy_loss, critic_loss, entropy, grad_norm, mean_action_ratio = optimize_model(
                agent, obss, actions, rewards, dones, old_log_probs, HP
            )
        log.add_scalar("loss/policy", policy_loss, global_step)
        log.add_scalar("loss/critic", critic_loss, global_step)
        log.add_scalar("loss/entropy", entropy, global_step)
        log.add_scalar("train/grad_norm", grad_norm, global_step)
        log.add_scalar("train/lr", optim.param_groups[0]["lr"], global_step)
        log.add_scalar("train/action_ratios", mean_action_ratio, global_step)

        global_step += actions.numel()

    log.close()
    envs.close()

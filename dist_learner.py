import queue
from dataclasses import dataclass
import envpool
import torch
import numpy as np
from torch import Tensor
from torch.distributions import Categorical
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from agent import Agent
from cli_params import CLIParams

torch.set_float32_matmul_precision("high")


class DistSettings:
    # Network.
    HOST = "127.0.0.1"
    PORT = 6767

    # Env.
    ENV_NAME = "Pong-v5"
    N_FRAME_STACK = 4
    OBS_SHAPE = (N_FRAME_STACK, 84, 84)
    
    ROLLOUT_STEPS = 32


@dataclass
class HyperParams(CLIParams):
    max_policy_lag: int = 8

    # Optim.
    lr: float = 1e-3
    adam_beta1: float = 0.1
    adam_beta2: float = 0.5

    # Training.
    train_steps: int = 5_000_000
    update_steps: int = 2
    batch_rollouts: int = 1

    # Impala.
    discount_gamma: float = 0.99
    impala_max_rho: float = 1.0
    impala_max_c: float = 1.0
    critic_loss_coeff: float = 0.5
    entropy_coeff: float = 0.01
    max_grad_norm: float = 0.5

    # Meta.
    device: str = "cuda"
    output_dir: str = "runs/test"


def receive_rollouts(rollout_queue: mp.Queue, policy_version: int, HP: HyperParams):
    batches = None
    n_batches = 0
    stale_rollouts = 0
    while n_batches < HP.batch_rollouts:
        batch = rollout_queue.get()
        if policy_version - batch["policy_version"] > HP.max_policy_lag:
            stale_rollouts += 1
            continue
        if batches is None:
            batches = {k: [] for k in batch}
        for k in batch:
            batches[k].append(batch[k])
        n_batches += 1

    actor_policy_version, total_reward, n_episodes = [np.array(batches[k]) for k in ("policy_version", "total_reward", "n_episodes")]
    actor_policy_version = actor_policy_version.mean()
    n_episodes = n_episodes.sum()
    mean_reward = None if n_episodes == 0 else total_reward.sum() / n_episodes

    obss, dones, rewards, old_log_probs = [
        torch.tensor(np.concat(batches[k], axis=1), dtype=torch.float32, device=HP.device)
        for k in ("obss", "dones", "rewards", "old_log_probs")
    ]
    actions = torch.tensor(np.concat(batches["actions"], axis=1), dtype=torch.int64, device=HP.device)
    return actor_policy_version, mean_reward, n_episodes, obss, dones, actions, rewards, old_log_probs, stale_rollouts


def optimize_model(
        agent: Agent, optim: Adam, obss: Tensor, actions: Tensor, rewards: Tensor, dones: Tensor, old_log_probs: Tensor, HP: HyperParams
    ) -> tuple[float, float, float, float, float]:
    # import matplotlib.pyplot as plt
    # for t in range(obss.shape[0]):
    #     fig, axs = plt.subplots(ncols=HP.n_frame_stack)
    #     for f in range(HP.n_frame_stack):
    #         axs[f].imshow(obss.cpu().numpy()[t, 0, f], cmap="gray")
    #     plt.show()
    logits, values = agent(obss.reshape(-1, *obss.shape[2:]))
    logits, values = [t.view(-1, obss.shape[1], *t.shape[1:]) for t in (logits, values)]
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
    for t in reversed(range(rho.shape[0])):
        vtrace_deltas[t] = rho[t] * td_deltas[t] + next_nonterminal[t] * HP.discount_gamma * c[t] * vtrace_deltas[t+1]
    vtrace_targets = vtrace_deltas + values

    advantages = rewards + next_nonterminal * HP.discount_gamma * vtrace_targets[1:] - values[:-1]

    return vtrace_targets, advantages, rho, action_ratios


if __name__ == "__main__":
    HP = HyperParams()

    # Create agent.
    envs = envpool.make_gymnasium(DistSettings.ENV_NAME, num_envs=1)  # TODO: make this exactly like actor env?
    agent = Agent(DistSettings.N_FRAME_STACK, envs.action_space.n).to(device=HP.device)
    optim = Adam(agent.parameters(), lr=HP.lr, betas=(HP.adam_beta1, HP.adam_beta2), fused=True)
    policy_version = -1

    log = SummaryWriter(log_dir=HP.output_dir)
    global_step = 0
    while global_step < HP.train_steps:
        policy_version += 1
        new_weights = (policy_version, {k: v.detach().cpu() for k, v in agent.state_dict().items()})
        for weight_queue in weight_queues:
            push_replace_queue(new_weights, weight_queue)

        (actor_policy_version, mean_reward, n_episodes, obss, dones,
         actions, rewards, old_log_probs, stale_rollouts) = receive_rollouts(rollout_queue, policy_version, HP)
        if n_episodes > 0:
            log.add_scalar("ep_stats/reward", mean_reward, global_step)
        log.add_scalar("ep_stats/episodes", n_episodes, global_step)
        log.add_scalar("async/policy_lag", policy_version - actor_policy_version, global_step)
        log.add_scalar("async/stale_rollouts", stale_rollouts, global_step)
        log.add_scalar("async/rollout_queue_len", rollout_queue.qsize(), global_step)
        print(f"{global_step}: reward={mean_reward}")

        # Linearly decay lr to 0.
        optim.param_groups[0]["lr"] = HP.lr - HP.lr * (global_step / HP.train_steps)

        # Optimize.
        for _ in range(HP.update_steps):
            policy_loss, critic_loss, entropy, grad_norm, mean_action_ratio = optimize_model(
                agent, optim, obss, actions, rewards, dones, old_log_probs, HP
            )
        log.add_scalar("loss/policy", policy_loss, global_step)
        log.add_scalar("loss/critic", critic_loss, global_step)
        log.add_scalar("loss/entropy", entropy, global_step)
        log.add_scalar("train/grad_norm", grad_norm, global_step)
        log.add_scalar("train/lr", optim.param_groups[0]["lr"], global_step)
        log.add_scalar("train/action_ratios", mean_action_ratio, global_step)

        global_step += actions.numel()

    # Cleanup.
    envs.close()
    log.close()
    print("GOODBYE from learner.")

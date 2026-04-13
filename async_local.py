"""IMPALA: https://arxiv.org/pdf/1802.01561."""
import multiprocessing as mp
from multiprocessing.synchronize import Event as EventClass
from dataclasses import dataclass
from queue import Full, Empty
import envpool
import torch
import numpy as np
from torch import Tensor
from torch.distributions import Categorical
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from cli_params import CLIParams
from agent import Agent
from env import EnvPoolEpisodeStats

torch.set_float32_matmul_precision("high")


@dataclass
class HyperParams(CLIParams):
    # Actor.
    n_actors: int = 4
    n_rollout_steps: int = 128
    rollout_queue_size: int = 32
    max_policy_lag: int = 8

    # Env.
    env_name: str = "Pong-v5"
    n_envs: int = 16
    n_frame_stack: int = 4

    # Training.
    n_epochs: int = 16000
    lr: float = 1e-3
    update_steps: int = 1

    discount_gamma: float = 0.99
    impala_max_rho: float = 1.0
    impala_max_c: float = 1.0
    critic_loss_coeff: float = 0.5
    entropy_coeff: float = 0.01
    max_grad_norm: float = 0.5

    # Meta.
    device: str = "cuda"
    output_dir: str = "runs/test"


def actor_func(actor_id: int, HP: HyperParams, rollout_queue: mp.Queue, weight_queue: mp.Queue, stop_event: EventClass):
    rollout_queue.cancel_join_thread()  # Don't wait for queue to flush.
    weight_queue.cancel_join_thread()
    print(f"HELLO from actor {actor_id}.")

    # Create envs.
    envs = envpool.make_gymnasium(HP.env_name, num_envs=HP.n_envs)
    envs = EnvPoolEpisodeStats(envs, n_envs=HP.n_envs)

    # Create agent.
    agent = Agent(HP.n_frame_stack, envs.action_space.n)
    policy_version, state_dict = weight_queue.get()
    agent.load_state_dict(state_dict)
    agent.to(device=HP.device).eval()

    # Storage buffers.
    OBS_SHAPE = (HP.n_frame_stack, 84, 84)
    obss = torch.zeros((HP.n_rollout_steps+1, HP.n_envs) + OBS_SHAPE, dtype=torch.float32, device=HP.device)
    dones = torch.zeros((HP.n_rollout_steps+1, HP.n_envs), dtype=torch.float32, device=HP.device)

    actions = np.zeros((HP.n_rollout_steps, HP.n_envs), dtype=np.int64)
    rewards = np.zeros((HP.n_rollout_steps, HP.n_envs), dtype=np.float32)
    old_log_probs = np.zeros((HP.n_rollout_steps, HP.n_envs), dtype=np.float32)

    # Env init.
    obs, _ = envs.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=HP.device)
    done = torch.zeros(HP.n_envs, dtype=torch.float32, device=HP.device)

    while not stop_event.is_set():
        policy_version = sync_weights(agent, weight_queue, policy_version, stop_event, HP.device)

        obs, done, mean_reward, mean_episode_length, n_episodes = sample_trajectories(
            agent, envs, HP.n_rollout_steps, obs, done, obss, dones, actions, rewards, old_log_probs
        )
        rollout = (actor_id, policy_version, mean_reward, mean_episode_length, n_episodes,
                   obss.cpu().to(dtype=torch.uint8).numpy(), dones.cpu().numpy(), actions, rewards, old_log_probs)
        push_replace_queue(rollout, rollout_queue)

    envs.close()
    print(f"GOODBYE from actor {actor_id}.")


def sync_weights(agent: Agent, weight_queue: mp.Queue, policy_version: int, stop_event: EventClass, device: str) -> int:
    if stop_event.is_set():
        return policy_version

    try:
        learner_policy_version, state_dict = weight_queue.get(block=False)
    except Empty:
        return policy_version

    while not stop_event.is_set():
        try:
            learner_policy_version, state_dict = weight_queue.get(block=False)
        except Empty:
            break

    if stop_event.is_set() or (learner_policy_version == policy_version):
        return policy_version

    agent.load_state_dict(state_dict)
    agent.to(device=device).eval()
    return learner_policy_version


def push_replace_queue(payload, queue: mp.Queue):
    try:
        queue.put(payload, block=False)
    except Full:
        try:
            queue.get(block=False)
        except Empty:
            pass
        try:
            queue.put(payload, block=False)
        except Full:
            pass


@torch.no_grad()
def sample_trajectories(
        agent: Agent, envs, n_rollout_steps: int, obs: Tensor, done: Tensor,
        obss: Tensor, dones: Tensor, actions: np.ndarray, rewards: np.ndarray, old_log_probs: np.ndarray,
    ) -> tuple[Tensor, Tensor, float | None, float | None, int]:
    total_reward = 0.0
    total_episode_length = 0
    n_episodes = 0

    # obs and done continue from prev epoch
    obss[0] = obs
    dones[0] = done

    for t in range(n_rollout_steps):
        # import matplotlib.pyplot as plt
        # plt.imshow(obs[0][0].cpu().numpy(), cmap="gray")
        # plt.show()
        logits = agent.get_action_logits(obs)
        action_dist = Categorical(logits=logits)
        action = action_dist.sample()

        actions[t] = action.cpu().numpy()
        old_log_probs[t] = action_dist.log_prob(action).cpu().numpy()

        obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
        rewards[t] = reward

        obs = torch.tensor(obs, dtype=torch.float32, device=obss.device)
        done = torch.tensor(terminated | truncated, dtype=torch.float32, device=dones.device)
        obss[t+1] = obs
        dones[t+1] = done

        if "stats" in info:
            done_mask = info["stats"]["done_mask"]
            total_reward += info["stats"]["returns"][done_mask].sum()
            total_episode_length += info["stats"]["lens"][done_mask].sum()
            n_episodes += int(done_mask.sum())

    mean_reward = None if n_episodes == 0 else total_reward / n_episodes
    mean_episode_length = None if n_episodes == 0 else total_episode_length / n_episodes
    return obs, done, mean_reward, mean_episode_length, n_episodes


def optimize_model(
        agent: Agent, optim: Adam, obss: Tensor, actions: Tensor, rewards: Tensor, dones: Tensor, old_log_probs: Tensor, HP: HyperParams
    ) -> tuple[float, float, float, float, float]:
    logits, values = agent(obss.reshape(-1, *obss.shape[2:]))
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

    # Create actors.
    mp.set_start_method("spawn")
    rollout_queue = mp.Queue(HP.rollout_queue_size)  # TODO: circular buffer?
    stop_event = mp.Event()
    weight_queues = [mp.Queue(1) for _ in range(HP.n_actors)]
    actors = [
        mp.Process(target=actor_func, args=(i, HP, rollout_queue, weight_queues[i], stop_event))
        for i in range(HP.n_actors)
    ]
    for actor in actors:
        actor.start()

    # Create agent.
    envs = envpool.make_gymnasium(HP.env_name, num_envs=1)  # TODO: make this exactly like actor env?
    agent = Agent(HP.n_frame_stack, envs.action_space.n).to(device=HP.device)
    optim = Adam(agent.parameters(), lr=HP.lr, fused=True)
    policy_version = -1

    log = SummaryWriter(log_dir=HP.output_dir)
    for epoch in range(HP.n_epochs):
        policy_version += 1
        new_weights = (policy_version, {k: v.detach().cpu() for k, v in agent.state_dict().items()})
        for weight_queue in weight_queues:
            push_replace_queue(new_weights, weight_queue)

        while True:
            (actor_id, actor_policy_version, mean_reward, mean_episode_length, n_episodes,
            obss, dones, actions, rewards, old_log_probs) = rollout_queue.get()
            if policy_version - actor_policy_version <= HP.max_policy_lag:
                break
        obss, dones, rewards, old_log_probs = [torch.tensor(t, dtype=torch.float32, device=HP.device)
                                               for t in (obss, dones, rewards, old_log_probs)]
        actions = torch.tensor(actions, dtype=torch.int64, device=HP.device)

        if n_episodes > 0:
            log.add_scalar("ep_stats/reward", mean_reward, epoch)
            log.add_scalar("ep_stats/length", mean_episode_length, epoch)
        log.add_scalar("ep_stats/episodes", n_episodes, epoch)
        log.add_scalar("async/policy_lag", policy_version - actor_policy_version, epoch)
        log.add_scalar("async/rollout_queue_len", rollout_queue.qsize(), epoch)
        print(f"{epoch}: reward={mean_reward}")

        # Linearly decay lr to 0.
        optim.param_groups[0]["lr"] = HP.lr - (HP.lr / HP.n_epochs) * epoch

        # Optimize.
        for _ in range(HP.update_steps):
            policy_loss, critic_loss, entropy, grad_norm, mean_action_ratio = optimize_model(
                agent, optim, obss, actions, rewards, dones, old_log_probs, HP
            )
        log.add_scalar("loss/policy", policy_loss, epoch)
        log.add_scalar("loss/critic", critic_loss, epoch)
        log.add_scalar("loss/entropy", entropy, epoch)
        log.add_scalar("train/grad_norm", grad_norm, epoch)
        log.add_scalar("train/lr", optim.param_groups[0]["lr"], epoch)
        log.add_scalar("train/action_ratios", mean_action_ratio, epoch)

    # Cleanup.
    envs.close()
    log.close()
    stop_event.set()
    print("STOP issued, waiting for actors to terminate.")
    for actor in actors:
        actor.join()
    print("DONE waiting for actors to terminate.")
    print("GOODBYE from learner.")

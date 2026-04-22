import time
import queue
import threading
import socket
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
from dist_log import DistLog
from dist_network import recv_msg, send_msg, MessageType
from dist_settings import DistSettings

torch.set_float32_matmul_precision("high")


@dataclass
class HyperParams(CLIParams):
    # Optim.
    lr: float = 1e-3
    adam_beta1: float = 0.1
    adam_beta2: float = 0.5

    # Training.
    train_steps: int = 5_000_000
    update_steps: int = 2
    batch_rollouts: int = 1
    max_policy_lag: int = 8

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


latest_weights = None
policy_version = -1
latest_weights_lock = threading.Lock()


def conn_handler(host: str, port: int, rollout_queue: queue.Queue):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen()
    while True:
        conn, addr = sock.accept()
        thread = threading.Thread(target=actor_handler, args=(conn, addr, rollout_queue), daemon=True)
        thread.start()


def actor_handler(conn: socket.socket, addr, rollout_queue: queue.Queue):
    print(f"CONNECTED to actor: {addr}.")
    try:
        while True:
            msg, _ = recv_msg(conn)
            if msg["type"] == MessageType.GET_WEIGHTS:
                while policy_version == -1:
                    time.sleep(0.1)
                if msg["policy_version"] == policy_version:
                    send_msg(conn, {"type": MessageType.WEIGHTS, "policy_version": policy_version})
                else:
                    with latest_weights_lock:
                        send_msg(conn, {"type": MessageType.WEIGHTS, "policy_version": policy_version, "state_dict": latest_weights})
            elif msg["type"] == MessageType.ROLLOUT:
                msg.pop("type")
                rollout_queue.put(msg)
                send_msg(conn, {"type": MessageType.ACK})
            else:
                raise ValueError(f"Unknown message type: {msg['type']}.")
    except ConnectionError:
        print(f"DISCONNECTED: {addr}")
    finally:
        conn.close()


def get_rollouts(rollout_queue: queue.Queue, policy_version: int, HP: HyperParams):
    batches = None
    n_batches = 0
    stale_rollouts = 0
    rollout_queue_get_time = 0.0
    while n_batches < HP.batch_rollouts:
        t0 = time.perf_counter()
        batch = rollout_queue.get()
        rollout_queue_get_time += time.perf_counter() - t0
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

    packed_obss, reset_prefixes, dones, actions, rewards, old_log_probs = [
        torch.tensor(np.concat(batches[k], axis=1), dtype=dtype, device=HP.device)
        for k, dtype in (("obss", torch.float32), ("reset_prefixes", torch.float32), ("dones", torch.bool),
                         ("actions", torch.int64), ("rewards", torch.float32), ("old_log_probs", torch.float32))
    ]
    latest_frames = packed_obss[DistSettings.N_FRAME_STACK:]
    obss = torch.empty((latest_frames.shape[0] + 1, packed_obss.shape[1], DistSettings.N_FRAME_STACK, *packed_obss.shape[2:]),
                       dtype=packed_obss.dtype, device=packed_obss.device)
    obss[0] = packed_obss[:DistSettings.N_FRAME_STACK].permute(1, 0, 2, 3)
    for t in range(latest_frames.shape[0]):
        obss[t+1, :, :-1] = obss[t, :, 1:]
        obss[t+1, :, -1] = latest_frames[t]
        done_mask = dones[t+1]
        obss[t+1, done_mask, :-1] = reset_prefixes[t, done_mask]
    dones = dones.to(dtype=torch.float32)
    return actor_policy_version, mean_reward, n_episodes, obss, dones, actions, rewards, old_log_probs, stale_rollouts, rollout_queue_get_time


def optimize_model(
        agent: Agent, optim: Adam, obss: Tensor, actions: Tensor, rewards: Tensor, dones: Tensor, old_log_probs: Tensor, HP: HyperParams
    ) -> tuple[float, float, float, float, float]:
    # import matplotlib.pyplot as plt
    # for t in range(obss.shape[0]):
    #     _, axs = plt.subplots(ncols=DistSettings.N_FRAME_STACK)
    #     for f in range(DistSettings.N_FRAME_STACK):
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

    rollout_queue = queue.Queue(maxsize=32)
    conn_thread = threading.Thread(target=conn_handler, args=(DistSettings.LISTEN_HOST, DistSettings.PORT, rollout_queue), daemon=True)
    conn_thread.start()

    envs = envpool.make_gymnasium(DistSettings.ENV_NAME, num_envs=1, stack_num=DistSettings.N_FRAME_STACK)  # TODO: make this exactly like actor env?
    agent = Agent(DistSettings.N_FRAME_STACK, envs.action_space.n).to(device=HP.device)
    optim = Adam(agent.parameters(), lr=HP.lr, betas=(HP.adam_beta1, HP.adam_beta2), fused=True)

    log = SummaryWriter(log_dir=HP.output_dir)
    console_log = DistLog()
    global_step = 0
    while global_step < HP.train_steps:
        t0 = time.perf_counter()
        with latest_weights_lock:
            policy_version += 1
            latest_weights = {k: v.detach().cpu().numpy() for k, v in agent.state_dict().items()}
        t1 = time.perf_counter()

        (actor_policy_version, mean_reward, n_episodes, obss, dones,
         actions, rewards, old_log_probs, stale_rollouts, rollout_queue_get_time) = get_rollouts(rollout_queue, policy_version, HP)
        t2 = time.perf_counter()
        if n_episodes > 0:
            log.add_scalar("ep_stats/reward", mean_reward, global_step)
        log.add_scalar("ep_stats/episodes", n_episodes, global_step)
        log.add_scalar("async/policy_lag", policy_version - actor_policy_version, global_step)
        log.add_scalar("async/stale_rollouts", stale_rollouts, global_step)
        log.add_scalar("async/rollout_queue_len", rollout_queue.qsize(), global_step)

        # Linearly decay lr to 0.
        optim.param_groups[0]["lr"] = HP.lr - HP.lr * (global_step / HP.train_steps)

        # Optimize.
        t3 = time.perf_counter()
        for _ in range(HP.update_steps):
            policy_loss, critic_loss, entropy, grad_norm, mean_action_ratio = optimize_model(
                agent, optim, obss, actions, rewards, dones, old_log_probs, HP
            )
        t4 = time.perf_counter()
        log.add_scalar("loss/policy", policy_loss, global_step)
        log.add_scalar("loss/critic", critic_loss, global_step)
        log.add_scalar("loss/entropy", entropy, global_step)
        log.add_scalar("train/grad_norm", grad_norm, global_step)
        log.add_scalar("train/lr", optim.param_groups[0]["lr"], global_step)
        log.add_scalar("train/action_ratios", mean_action_ratio, global_step)

        global_step += actions.numel()

        # Log.
        total_time = t4 - t0
        times = "  ".join([
            console_log.pct("wgt_sync", (t1 - t0) / total_time),
            console_log.pct("roll_q", rollout_queue_get_time / total_time),
            console_log.pct("prep_batch", (t2 - t1 - rollout_queue_get_time) / total_time),
            console_log.pct("optim", (t4 - t3) / total_time),
        ])
        reward_text = f"reward {mean_reward:7.2f}" if mean_reward is not None else " " * len("reward    0.00")
        print(f"{global_step}: {reward_text}  {times}  {console_log.scalar('freq', 1.0 / total_time)}")

    # Cleanup.
    envs.close()
    log.close()
    print("GOODBYE from learner.")

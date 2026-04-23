import time
import socket
import uuid
from argparse import ArgumentParser
import envpool
import torch
import numpy as np
from torch.distributions import Categorical
from agent import Agent
from dist_log import DistLog
from env import EnvPoolEpisodeStats
from dist_network import send_msg, recv_msg, MessageType
from dist_settings import DistSettings


ROLLOUT_STEPS = 32
N_ENVS = 16


def get_weights(sock: socket.socket, agent: Agent, actor_policy_version: int, device: str) -> tuple[int, int]:
    send_msg(sock, {"type": MessageType.GET_WEIGHTS, "policy_version": actor_policy_version})
    msg, msg_size = recv_msg(sock)
    if msg["policy_version"] == actor_policy_version:
        return actor_policy_version, msg_size

    state_dict = {k: torch.tensor(v, device=device) for k, v in msg["state_dict"].items()}
    agent.load_state_dict(state_dict)
    agent.eval()
    return msg["policy_version"], msg_size


@torch.no_grad()
def sample_trajectories(
        agent: Agent, envs, rollout_steps: int, obs: np.ndarray, done: np.ndarray, obss: np.ndarray,
        dones: np.ndarray, actions: np.ndarray, rewards: np.ndarray, old_log_probs: np.ndarray, device: str,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
    total_reward = 0.0
    n_episodes = 0

    # obs and done continue from prev samples.
    obss[0] = obs
    dones[0] = done

    for t in range(rollout_steps):
        # import matplotlib.pyplot as plt
        # plt.imshow(obs[0][0], cmap="gray")
        # plt.show()
        logits = agent.get_action_logits(torch.tensor(obs, device=device))
        action_dist = Categorical(logits=logits)
        action = action_dist.sample()
        old_log_probs[t] = action_dist.log_prob(action).cpu().numpy()

        action = action.cpu().numpy()
        actions[t] = action

        obs, reward, terminated, truncated, info = envs.step(action)
        rewards[t] = reward

        done = terminated | truncated
        obss[t+1] = obs
        dones[t+1] = done

        if "stats" in info:
            done_mask = info["stats"]["done_mask"]
            total_reward += float(info["stats"]["returns"][done_mask].sum())
            n_episodes += int(done_mask.sum())
    return obs, done, total_reward, n_episodes


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    actor_id = uuid.uuid4().hex
    print(f"HELLO from actor: {actor_id}. device={args.device}")

    # Connect to learner.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while True:
        try:
            sock.connect((DistSettings.LEARNER_HOST, DistSettings.PORT))
            break
        except ConnectionRefusedError:
            time.sleep(0.1)

    # Create envs.
    envs = envpool.make_gymnasium(DistSettings.ENV_NAME, num_envs=N_ENVS, stack_num=DistSettings.N_FRAME_STACK)
    envs = EnvPoolEpisodeStats(envs, n_envs=N_ENVS)

    # Create agent.
    agent = Agent(DistSettings.N_FRAME_STACK, envs.action_space.n).to(device=args.device)
    policy_version, _ = get_weights(sock, agent, -1, args.device)

    # Storage buffers.
    obss = np.zeros((ROLLOUT_STEPS+1, N_ENVS) + DistSettings.OBS_SHAPE, dtype=np.uint8)
    dones = np.zeros((ROLLOUT_STEPS+1, N_ENVS), dtype=np.bool)

    actions = np.zeros((ROLLOUT_STEPS, N_ENVS), dtype=np.int64)
    rewards = np.zeros((ROLLOUT_STEPS, N_ENVS), dtype=np.float32)
    old_log_probs = np.zeros((ROLLOUT_STEPS, N_ENVS), dtype=np.float32)

    # Env init.
    obs, _ = envs.reset()
    done = np.zeros(N_ENVS, dtype=np.bool)

    n_rollout = 0
    log = DistLog()
    while True:
        t0 = time.perf_counter()
        try:
            policy_version, weight_msg_size = get_weights(sock, agent, policy_version, args.device)
        except ConnectionError as exc:
            print(exc)
            break
        t1 = time.perf_counter()

        obs, done, total_reward, n_episodes = sample_trajectories(
            agent, envs, ROLLOUT_STEPS, obs, done, obss, dones, actions, rewards, old_log_probs, args.device
        )
        t2 = time.perf_counter()

        done_mask = dones[1:]
        reset_prefixes = np.zeros_like(obss[1:, :, :-1])
        reset_prefixes[done_mask] = obss[1:, :, :-1][done_mask]
        rollout = {"type": MessageType.ROLLOUT, "actor_id": actor_id, "policy_version": policy_version,
                   "total_reward": total_reward, "n_episodes": n_episodes,
                   "obss": np.concat((np.moveaxis(obss[0], 1, 0), obss[1:, :, -1]), axis=0),
                   "reset_prefixes": reset_prefixes, "dones": dones, "actions": actions,
                   "rewards": rewards, "old_log_probs": old_log_probs}
        t3 = time.perf_counter()

        try:
            rollout_msg_size = send_msg(sock, rollout)
            msg, _ = recv_msg(sock)
        except ConnectionError:
            break
        t4 = time.perf_counter()
        assert msg["type"] == MessageType.ACK
        n_rollout += 1

        # Log.
        total_time = t4 - t0
        times = "  ".join([
            log.pct("wgt_sync", (t1 - t0) / total_time),
            log.pct("sample", (t2 - t1) / total_time),
            log.pct("prep_roll", (t3 - t2) / total_time),
            log.pct("send_roll", (t4 - t3) / total_time),
        ])
        print(f"{n_rollout}:  {times}  {log.scalar('freq', 1.0 / total_time)}  ", end="")
        print(f"{log.kb('wgt_sz', weight_msg_size)}  {log.kb('roll_sz', rollout_msg_size)}")

    envs.close()
    sock.close()
    print(f"GOODBYE from actor: {actor_id}.")

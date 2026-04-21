import time
import socket
import uuid
import envpool
import torch
import numpy as np
from torch import Tensor
from torch.distributions import Categorical
from agent import Agent
from dist_log import DistLog
from env import EnvPoolEpisodeStats
from dist_network import send_msg, recv_msg, MessageType
from dist_settings import DistSettings


ROLLOUT_STEPS = 32
N_ENVS = 16


def get_weights(sock: socket.socket, agent: Agent, actor_policy_version: int) -> tuple[int, int]:
    send_msg(sock, {"type": MessageType.GET_WEIGHTS, "policy_version": actor_policy_version})
    msg, msg_size = recv_msg(sock)
    if msg["policy_version"] == actor_policy_version:
        return actor_policy_version, msg_size

    state_dict = {k: torch.tensor(v) for k, v in msg["state_dict"].items()}
    agent.load_state_dict(state_dict)
    agent.eval()
    return msg["policy_version"], msg_size


@torch.no_grad()
def sample_trajectories(
        agent: Agent, envs, rollout_steps: int, obs: Tensor, done: Tensor,
        obss: Tensor, dones: Tensor, actions: np.ndarray, rewards: np.ndarray, old_log_probs: np.ndarray,
    ) -> tuple[Tensor, Tensor, float, float]:
    total_reward = 0.0
    n_episodes = 0

    # obs and done continue from prev samples.
    obss[0] = obs
    dones[0] = done

    for t in range(rollout_steps):
        # import matplotlib.pyplot as plt
        # plt.imshow(obs[0][0].cpu().numpy(), cmap="gray")
        # plt.show()
        logits = agent.get_action_logits(obs)
        action_dist = Categorical(logits=logits)
        action = action_dist.sample()

        actions[t] = action.numpy()
        old_log_probs[t] = action_dist.log_prob(action).numpy()

        obs, reward, terminated, truncated, info = envs.step(action.numpy())
        rewards[t] = reward

        obs = torch.tensor(obs, dtype=torch.float32)
        done = torch.tensor(terminated | truncated, dtype=torch.float32)
        obss[t+1] = obs
        dones[t+1] = done

        if "stats" in info:
            done_mask = info["stats"]["done_mask"]
            total_reward += float(info["stats"]["returns"][done_mask].sum())
            n_episodes += int(done_mask.sum())
    return obs, done, total_reward, n_episodes


if __name__ == "__main__":
    actor_id = uuid.uuid4().hex
    print(f"HELLO from actor: {actor_id}.")

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
    agent = Agent(DistSettings.N_FRAME_STACK, envs.action_space.n)
    policy_version = -1
    policy_version, _ = get_weights(sock, agent, policy_version)

    # Storage buffers.
    obss = torch.zeros((ROLLOUT_STEPS+1, N_ENVS) + DistSettings.OBS_SHAPE, dtype=torch.float32)
    dones = torch.zeros((ROLLOUT_STEPS+1, N_ENVS), dtype=torch.float32)

    actions = np.zeros((ROLLOUT_STEPS, N_ENVS), dtype=np.int64)
    rewards = np.zeros((ROLLOUT_STEPS, N_ENVS), dtype=np.float32)
    old_log_probs = np.zeros((ROLLOUT_STEPS, N_ENVS), dtype=np.float32)

    # Env init.
    obs, _ = envs.reset()
    obs = torch.tensor(obs, dtype=torch.float32)
    done = torch.zeros(N_ENVS, dtype=torch.float32)

    n_rollout = 0
    log = DistLog()
    while True:
        t0 = time.perf_counter()
        try:
            policy_version, weight_msg_size = get_weights(sock, agent, policy_version)
        except ConnectionError as exc:
            print(exc)
            break
        t1 = time.perf_counter()

        obs, done, total_reward, n_episodes = sample_trajectories(
            agent, envs, ROLLOUT_STEPS, obs, done, obss, dones, actions, rewards, old_log_probs
        )
        t2 = time.perf_counter()

        rollout = {"type": MessageType.ROLLOUT, "actor_id": actor_id, "policy_version": policy_version,
                "total_reward": total_reward, "n_episodes": n_episodes,
                "obss": obss.to(dtype=torch.uint8).numpy(),
                "dones": dones.numpy(), "actions": actions,
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

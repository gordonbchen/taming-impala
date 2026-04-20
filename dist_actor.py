import time
import socket
import uuid
from argparse import ArgumentParser
import envpool
import torch
import numpy as np
from torch import Tensor
from torch.distributions import Categorical
from agent import Agent
from env import EnvPoolEpisodeStats
from dist_network import send_msg, recv_msg
from dist_learner import DistSettings


def get_weights(sock: socket.socket, agent: Agent, actor_policy_version: int) -> tuple[int, bool]:
    send_msg(sock, {"type": "GET_WEIGHTS", "policy_version": actor_policy_version})
    msg = recv_msg(sock)
    if msg["type"] == "STOP":
        return actor_policy_version, True
    if msg["policy_version"] == actor_policy_version:
        return actor_policy_version, False
    agent.load_state_dict(msg["state_dict"])
    agent.eval()
    return msg["policy_version"], False


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
            total_reward += info["stats"]["returns"][done_mask].sum()
            n_episodes += int(done_mask.sum())
    return obs, done, total_reward, n_episodes


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_envs", type=int, default=16)
    args = parser.parse_args()

    actor_id = uuid.uuid4().hex
    print(f"HELLO from actor: {actor_id}.")

    # Connect to learner.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while True:
        try:
            sock.connect((DistSettings.HOST, DistSettings.PORT))
            break
        except ConnectionRefusedError:
            time.sleep(0.1)

    # Create envs.
    envs = envpool.make_gymnasium(DistSettings.ENV_NAME, num_envs=args.n_envs)
    envs = EnvPoolEpisodeStats(envs, n_envs=args.n_envs)

    # Create agent.
    agent = Agent(DistSettings.N_FRAME_STACK, envs.action_space.n)
    policy_version = -1
    policy_version, stop = get_weights(sock, agent, policy_version)

    # Storage buffers.
    obss = torch.zeros((DistSettings.ROLLOUT_STEPS+1, args.n_envs) + DistSettings.OBS_SHAPE, dtype=torch.float32)
    dones = torch.zeros((DistSettings.ROLLOUT_STEPS+1, args.n_envs), dtype=torch.float32)

    actions = np.zeros((DistSettings.ROLLOUT_STEPS, args.n_envs), dtype=np.int64)
    rewards = np.zeros((DistSettings.ROLLOUT_STEPS, args.n_envs), dtype=np.float32)
    old_log_probs = np.zeros((DistSettings.ROLLOUT_STEPS, args.n_envs), dtype=np.float32)

    # Env init.
    obs, _ = envs.reset()
    obs = torch.tensor(obs, dtype=torch.float32)
    done = torch.zeros(args.n_envs, dtype=torch.float32)

    while not stop:
        policy_version, stop = get_weights(sock, agent, policy_version)
        if stop: break

        obs, done, total_reward, n_episodes = sample_trajectories(
            agent, envs, DistSettings.ROLLOUT_STEPS, obs, done, obss, dones, actions, rewards, old_log_probs
        )
        rollout = {"type": "ROLLOUT", "actor_id": actor_id, "policy_version": policy_version,
                   "total_reward": total_reward, "n_episodes": n_episodes,
                   "obss": obss.to(dtype=torch.uint8).numpy(), "dones": dones.numpy(), "actions": actions,
                   "rewards": rewards, "old_log_probs": old_log_probs}
        send_msg(sock, rollout)

    envs.close()
    sock.close()
    print(f"GOODBYE from actor: {actor_id}.")

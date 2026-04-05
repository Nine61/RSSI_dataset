import numpy as np
import torch
from pathlib import Path
from env import DroneEnv
from ppo import PPO

BASE_DIR = Path(__file__).resolve().parent
BEST_MODEL_PATH = BASE_DIR / "best_ppo_drone.pth"
REWARDS_PATH = BASE_DIR / "rewards_history.npy"


def train():
    env = DroneEnv()
    agent = PPO(env.state_dim, env.action_dim)

    max_episodes = 3000
    update_timestep = 2000

    timestep = 0
    episode_rewards = []

    memory = {
        "states": [],
        "actions": [],
        "log_probs": [],
        "rewards": [],
        "dones": [],
        "values": [],
    }

    best_reward = -np.inf

    print("=== 드론 자율비행 PPO 학습 시작 ===")
    print(f"모델 저장 위치: {BEST_MODEL_PATH}")
    print(f"보상 저장 위치: {REWARDS_PATH}")

    for ep in range(1, max_episodes + 1):
        state = env.reset()
        ep_reward = 0

        while True:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            memory["states"].append(state)
            memory["actions"].append(action)
            memory["log_probs"].append(log_prob)
            memory["rewards"].append(reward)
            memory["dones"].append(done)
            memory["values"].append(value)

            state = next_state
            ep_reward += reward
            timestep += 1

            if timestep % update_timestep == 0:
                agent.update(memory)
                memory = {k: [] for k in memory.keys()}

            if done:
                break

        episode_rewards.append(ep_reward)

        if ep % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode: {ep}\t Avg Reward (Last 100): {avg_reward:.2f}\t Last Result: {info}")

            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save(agent.policy.state_dict(), BEST_MODEL_PATH)

    np.save(REWARDS_PATH, episode_rewards)

    print("=== 학습 완료 ===")
    print(f"best_ppo_drone.pth 생성 위치: {BEST_MODEL_PATH}")
    print(f"rewards_history.npy 생성 위치: {REWARDS_PATH}")


if __name__ == "__main__":
    train()
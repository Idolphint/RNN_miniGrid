import time
import gymnasium as gym
import numpy as np
import imageio
import datetime
import os
from PPO import PPO
from PPO_RNN import PPO_rnn
from envs import FetchReturnEnv, CrossingEnv


# env define
grid_size = 9
agent_view_size = 3
ac_name = "RNN_replay"
env_name = "Crossing"  # DoorKey Crossing FetchReturn
more_obs = False
state_dim = agent_view_size**2 * 3
if env_name == "DoorKey":
    state_dim += 3
    env = gym.make(f"MiniGrid-DoorKey-{grid_size}x{grid_size}-v0", render_mode="rgb_array", agent_view_size=agent_view_size)
elif env_name == "FetchReturn":
    state_dim += 4+2+1+7
    env = FetchReturnEnv(size=grid_size, render_mode="rgb_array", gen_obstacle=more_obs, agent_view_size=agent_view_size)
elif env_name == "Crossing":
    state_dim += 4+2+1+7
    env = CrossingEnv(size=grid_size, render_mode="rgb_array", agent_view_size=agent_view_size, max_steps=300)


if ac_name == "simpleNN":
    ppo_agent = PPO(state_dim=state_dim, action_dim=7)
    ppo_agent.load("./weights/ppo.pt")
elif ac_name == "RNN":
    ppo_agent = PPO_rnn(state_dim=state_dim, action_dim=7, ac_name="RNN")
    ppo_agent.load("./weights/ppo_RNN_Crossinggrid11_2024-07-30.pt")
    hidden = ppo_agent.policy.init_hidden()
elif ac_name == "RNN_replay":
    ppo_agent = PPO_rnn(state_dim=state_dim, action_dim=7, ac_name="RNN")
    # ppo_agent.load("./weights/ppo_RNN_grid5_2024-07-17.pt")
    ppo_agent.load("./weights/ppo_RNN_Crossinggrid9_2024-07-30.pt")
    hidden = ppo_agent.policy.init_hidden()

observation, info = env.reset(seed=42)

frames = []
for t in range(500):
    # print(observation)
    if env_name == "DoorKey":
        img = observation["image"].reshape(agent_view_size**2, 3)
        dir = observation["direction"]
        state = np.zeros([agent_view_size**2+1, 3])
        state[49, 0] = dir
        state[:49, :] = img
    elif env_name == "FetchReturn" or env_name == "Crossing":
        img = observation["image"].reshape(agent_view_size**2, 3)
        other_info = np.zeros(4 + 2 + 1 + 7)
        other_info[observation["direction"]] = 1  # one-hot dir
        other_info[observation["has_goal"] + 4] = 1  # one-hot hand goal

        state = np.zeros(state_dim)
        state[:agent_view_size**2 * 3] = img.reshape(-1)
        state[agent_view_size**2 * 3:] = other_info


    if ac_name == "simpleNN":
        action = ppo_agent.select_action(state)  # User-defined policy function
        observation, reward, terminated, truncated, info = env.step(action)
        ppo_agent.buffer.rewards.append(reward)
        ppo_agent.buffer.is_terminals.append(terminated)
    elif ac_name == "RNN":
        action, hidden = ppo_agent.select_action(state, hidden)
        observation, reward, terminated, truncated, info = env.step(action)
        ppo_agent.buffer.rewards.append(reward)
        ppo_agent.buffer.is_terminals.append(terminated)
    elif ac_name == "RNN_replay":
        action, observation, reward, terminated, truncated = ppo_agent.play(env, state, test=True)
    print("step ", t, action.item(), reward, terminated, truncated, observation["has_goal"])
    frame = env.render()
    frames.append(frame)
    # time.sleep(0.3)
    if terminated or truncated:
        # epi += 1
        observation, info = env.reset()
        frame = env.render()
        frames.append(frame)
        if ac_name == "RNN":
            hidden = ppo_agent.policy.init_hidden()

video_path = f'./video/{grid_size}x{grid_size}grid_{env_name}obs{more_obs}_{ac_name}_{datetime.datetime.today().strftime("%m-%d-%H")}.gif'
imageio.mimsave(video_path, frames, fps=8)
env.close()
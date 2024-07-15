import time
import gymnasium as gym
import numpy as np
import imageio
import datetime
import os
from PPO import PPO
from PPO_RNN import PPO_rnn

# env define
grid_size = 8
env = gym.make(f"MiniGrid-DoorKey-{grid_size}x{grid_size}-v0", render_mode="rgb_array")
ac_name = "RNN"
if ac_name == "simpleNN":
    ppo_agent = PPO(state_dim=150, action_dim=7)
    ppo_agent.load("./weights/ppo.pt")
elif ac_name == "RNN":
    ppo_agent = PPO_rnn(state_dim=150, action_dim=7, ac_name="RNN")
    ppo_agent.load("./weights/ppo_RNN_grid8.pt")
    hidden = ppo_agent.policy.init_hidden()
observation, info = env.reset(seed=42)

frames = []
for t in range(100):
    # print(observation)
    img = observation["image"].reshape(7*7, 3)
    dir = observation["direction"]
    state = np.zeros([7*7+1, 3])
    state[49, 0] = dir
    state[:49, :] = img
    if ac_name == "simpleNN":
        action = ppo_agent.select_action(state)  # User-defined policy function
    elif ac_name == "RNN":
        action, hidden = ppo_agent.select_action(state, hidden)
    observation, reward, terminated, truncated, info = env.step(action)

    ppo_agent.buffer.rewards.append(reward)
    ppo_agent.buffer.is_terminals.append(terminated)

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

imageio.mimsave(f'{grid_size}x{grid_size}grid_doorKey_{ac_name}_{datetime.datetime.today().strftime("%m-%d-%H")}.gif', frames, fps=8)
env.close()
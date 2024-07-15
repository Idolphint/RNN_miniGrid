# 定义一个miniGrid中的开门环境
import gymnasium as gym
from policy import BasePolicy

# class DoorKey:
#     def __init__(self, size=5):
#         self.env = gym.make(f"MiniGrid-DoorKey-{size}x{size}-v0", render_mode="human")
env = gym.make(f"MiniGrid-DoorKey-5x5-v0", render_mode="human")
policy = BasePolicy()
observation, info = env.reset(seed=42)
env.render()
for _ in range(100):
    action = policy.get_action(observation)  # User-defined policy function
    observation, reward, terminated, truncated, info = env.step(action)
    print(reward)
    env.render()
    if terminated or truncated:
        observation, info = env.reset()
env.close()
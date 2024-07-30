import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from policy import SimpleNN
import datetime
import os
# from PPO import PPO
from PPO_RNN import PPO_rnn
from envs import FetchReturnEnv, CrossingEnv

# env define
grid_size = 9
agent_view_size = 3
ac_name = "RNN"
env_name = "Crossing"  # DoorKey Crossing FetchReturn
more_obs = False
if env_name == "DoorKey":
    env = gym.make(f"MiniGrid-DoorKey-{grid_size}x{grid_size}-v0", render_mode="rgb_array", agent_view_size=agent_view_size)
elif env_name == "FetchReturn":
    env = FetchReturnEnv(size=grid_size, render_mode="rgb_array", gen_obstacle=more_obs, agent_view_size=agent_view_size)
elif env_name == "Crossing":
    env = CrossingEnv(size=grid_size, render_mode="rgb_array", agent_view_size=agent_view_size, max_steps=300)
# constant define
max_ep_len = 100
update_timestep = 4 * max_ep_len
actor_lr = 3e-4
critic_lr = 1e-3
lr_change_step = (100000, 1000000)
eps_clip = 0.2
gamma = 0.93

# log defines
log_dir = f"./log/PPO_{ac_name}_{env_name}{grid_size}{'_obs' if more_obs else ''}_" + datetime.datetime.now().strftime(
    '%Y-%m-%d_%H')
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

# agent define
state_dim = agent_view_size * agent_view_size * 3 + 4+2+1+7
ppo_agent = PPO_rnn(state_dim=state_dim, action_dim=7, ac_name=ac_name,
                    lr_actor=actor_lr, lr_critic=critic_lr, gamma=gamma, eps_clip=eps_clip)
ckpt_path = f"./weights/ppo_{ac_name}_{env_name}grid{grid_size}{'_obs' if more_obs else ''}_{datetime.datetime.now().strftime('%Y-%m-%d')}.pt"
load_path = "./weights/ppo_RNN_Crossinggrid9_2024-07-30.pt"

# init env and agent
if load_path is not None and os.path.exists(load_path):
    ppo_agent.load(load_path)
    print("loading ckpt from", load_path)


def train():
    epi = 0
    observation, info = env.reset()
    hidden = ppo_agent.policy.init_hidden()
    # tmp variance
    epi_reward = 0
    last_save_reward = -100
    terminated = True
    epi_begin_t = 0

    for t in range(4000000):
        # print(observation)
        img = observation["image"].reshape(agent_view_size * agent_view_size, 3)
        dir = observation["direction"]
        state = np.zeros([agent_view_size * agent_view_size + 1, 3])
        state[agent_view_size * agent_view_size, 0] = dir
        state[:agent_view_size * agent_view_size, :] = img
        # if terminated:
        #     # hidden = ppo_agent.policy.init_hidden(state.reshape(1, -1), set_cuda=True)
        #     hidden = ppo_agent.policy.init_hidden()
        action, hidden = ppo_agent.select_action(state, hidden)  # User-defined policy function
        observation, reward, terminated, truncated, info = env.step(action)
        if reward == 0:
            reward = -0.0001  # 如果负值太大，非常影响训练，需要尽可能小
        epi_reward = epi_reward * 0.99 + reward

        # log_f.write(f'{epi},{t},{reward}\n')
        # log_f.flush()
        writer.add_scalar("reward", reward, global_step=t)
        ppo_agent.buffer.rewards.append(reward)
        ppo_agent.buffer.is_terminals.append(terminated)

        if (t + 1) % update_timestep == 0:
            print(epi, t, epi_reward)
            ppo_agent.update(writer)
        if (t + 1) % 10000 == 0:
            if last_save_reward < epi_reward:
                ppo_agent.save(ckpt_path)
                last_save_reward = epi_reward
        if terminated or truncated:
            epi += 1
            observation, info = env.reset()
            hidden = ppo_agent.policy.init_hidden()

    env.close()
    writer.close()


def play_and_collect_data(writer, test=True, t=0):
    avg_reward = 0
    for i in range(ppo_agent.buffer.num_game_per_batch):
        terminated = False
        truncated = False
        observation, info = env.reset()
        while not (terminated or truncated):
            img = observation["image"].reshape(agent_view_size**2, 3)
            other_info = np.zeros(4+2+1+7)
            other_info[observation["direction"]] = 1  # one-hot dir
            other_info[observation["has_goal"]+4] = 1  # one-hot hand goal

            state = np.zeros(state_dim)
            state[:agent_view_size * agent_view_size*3] =img.reshape(-1)
            state[agent_view_size * agent_view_size*3:] = other_info

            action, observation, reward, terminated, truncated = ppo_agent.play(env, state, test)
            # print(t, ":  ", action.item(), reward)
            writer.add_scalar("reward", reward, global_step=t)
            t += 1

            avg_reward += reward
    return avg_reward / (ppo_agent.buffer.num_game_per_batch), t


def train_by_play():
    last_saved_reward = -100
    global_timestep = 0
    while global_timestep < int(10 ** grid_size):  # 100000
        avg_reward, global_timestep = play_and_collect_data(writer, test=False, t=global_timestep)
        print("avg_reward", avg_reward, global_timestep)
        if avg_reward > last_saved_reward:
            ppo_agent.save(ckpt_path)
            last_saved_reward = avg_reward
        ppo_agent.update_new(writer)

    writer.close()


if __name__ == '__main__':
    # train()
    train_by_play()

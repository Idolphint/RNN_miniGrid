# 定义一个拿完目标再最快返回的任务，环境要有一些障碍，size应当有大有小
import gymnasium as gym
from policy import BasePolicy

from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
import imageio
import random
import matplotlib.pyplot as plt
import numpy as np


def Simple_DoorKey():
    grid = 16
    env = gym.make(f"MiniGrid-DoorKey-{grid}x{grid}-v0", render_mode="human")
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


class FetchReturnEnv(MiniGridEnv):
    def __init__(self, size=5, render_mode="human", gen_obstacle=True):
        self.size = size
        self.mission = "fetch the goal and return"
        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=4 * size * size,
            see_through_walls=True,
            render_mode=render_mode
        )
        self.start_pos = (1, 1)  # should be random sample
        self.last_pos = (1, 1)
        self.goal_pos = (self.size - 2, self.size - 2)  # should be random
        self.has_goal = False
        self.gen_obstacles = gen_obstacle

    @staticmethod
    def _gen_mission():
        return "fetch the goal and return"

    def _gen_grid(self, width, height):
        # 创建空的网格
        self.grid = Grid(width, height)

        # 墙壁
        self.grid.wall_rect(0, 0, width, height)

        # sample一个起始点
        self._get_start_pose()

        # 随机游走一定步长确定goal点
        visit_map, self.goal_pos = self._get_obj_place(width, height)

        # 放置目标点
        self.put_obj(Goal(), *self.goal_pos)

        # 防止障碍，控制obs_num和obs_len使难度加大
        if self.gen_obstacles:
            self._get_obstacle(visit_map, obs_num=4, obs_len=4)

        # 智能体初始位置
        self.agent_pos = self.start_pos
        self.agent_dir = 0

    def _get_start_pose(self):
        k = np.random.randint(0, 4)
        if k == 0:
            self.start_pos = (np.random.randint(1, self.size - 1), 1)
        elif k == 1:
            self.start_pos = (np.random.randint(1, self.size - 1), self.size - 2)
        elif k == 2:
            self.start_pos = (1, np.random.randint(1, self.size - 1))
        elif k == 3:
            self.start_pos = (self.size - 2, np.random.randint(1, self.size - 1))

    def random_walk(self, visit_map, start_pos, max_step=1000, terminal_func=None):
        width, height = visit_map.shape
        end_pose = start_pos.copy()

        visit_map[end_pose] = 1

        s = 0
        while s < max_step:
            s += 1
            able_point = []
            if end_pose[0] + 1 < width and not visit_map[end_pose[0] + 1, end_pose[1]]:
                able_point.append([end_pose[0] + 1, end_pose[1]])
            if end_pose[1] + 1 < height and not visit_map[end_pose[0], end_pose[1] + 1]:
                able_point.append([end_pose[0], end_pose[1] + 1])
            if end_pose[0] - 1 >= 0 and not visit_map[end_pose[0] - 1, end_pose[1]]:
                able_point.append([end_pose[0] - 1, end_pose[1]])
            if end_pose[1] - 1 >= 0 and not visit_map[end_pose[0], end_pose[1] - 1]:
                able_point.append([end_pose[0], end_pose[1] - 1])

            if len(able_point) == 0:  # rand_walk end
                break
            else:
                next_pos = able_point[np.random.randint(low=0, high=len(able_point))]
            visit_map[next_pos[0], next_pos[1]] = 1
            end_pose = next_pos
            if terminal_func and terminal_func(s, end_pose, start_pos):
                break
        return visit_map, end_pose

    def _get_obj_place(self, width, height, max_tries=10):
        # 期望走一条通路，在这条通路上没有障碍，其他地方随机出现障碍，期望障碍是成块出现的，同样采用随机游走n步的方式生成
        # 通路希望
        goal_pos_ = self.start_pos
        visit_map = np.zeros((width - 2, height - 2))
        visit_map[goal_pos_[0] - 1, goal_pos_[1] - 1] = 1

        def end_walk_func(step, now_pos, start_pos):
            if step > (self.size - 2) and \
                    abs(now_pos[0] - start_pos[0]) + abs(now_pos[1] - start_pos[1]) > self.size:
                return True

        for _ in range(max_tries):
            visit_map = np.zeros((width - 2, height - 2))
            visit_map, end_pos = self.random_walk(visit_map, start_pos=[self.start_pos[0] - 1, self.start_pos[1] - 1],
                                                  max_step=self.max_steps, terminal_func=end_walk_func)
            goal_pos_ = (end_pos[0] + 1, end_pos[1] + 1)
            found_path = abs(goal_pos_[0] - self.start_pos[0]) + abs(goal_pos_[1] - self.start_pos[1]) > self.size - 4
            # print(found_path, visit_map, self.start_pos, end_pos)
            if found_path:
                break

        return visit_map, goal_pos_

    def _get_obstacle(self, visit_map, obs_num=0, obs_len=0):
        if obs_num == 0:
            obs_num = np.random.randint(1, (self.size - 2) // 2)
        if obs_len == 0:
            obs_len = np.random.randint(4)
        for o in range(obs_num):
            # first obs block
            no_visit = np.where(visit_map == 0)
            if len(no_visit) != 2 or len(no_visit[0]) == 0:
                print("no left to visit", no_visit)
                break
            idx = np.random.randint(len(no_visit))
            point = [no_visit[0][idx], no_visit[1][idx]]
            visit_map[point] = 1
            self.put_obj(Wall(), point[0] + 1, point[1] + 1)

            visit_map_after, _ = self.random_walk(visit_map.copy(), start_pos=point, max_step=obs_len)
            new_obs = np.where((visit_map_after == 1) & (visit_map == 0))
            for i in range(len(new_obs[0])):
                # print(new_obs[0][i], new_obs[1][i])
                self.put_obj(Wall(), new_obs[0][i] + 1, new_obs[1][i] + 1)
            visit_map = visit_map_after

    def step(self, action):
        obs, _, done, truncated, info = super().step(action)
        if self.has_goal:
            reward = -0.0001
        else:
            reward = -0.00001
        # print("ori reward truncated, info, done", reward, truncated, info, done)
        # 检查是否到达目标
        if tuple(self.agent_pos) == self.goal_pos and not self.has_goal:
            self.has_goal = True
            done = False
            truncated = False
            reward = 0.3 * super()._reward()  # 吃掉目标获得奖励
            self.grid.set(*self.goal_pos, None)
        # elif not self.has_goal:
        #     reward = -0.00001

        # 检查是否返回起点
        if self.has_goal and tuple(self.agent_pos) == self.start_pos:
            done = True
            reward = super()._reward()  # 返回起点获得额外奖励
        # elif self.has_goal:  # and self.agent_pos == self.last_pos:
        #     reward = -0.0001  # 拿到目标后需要尽快移动以找到出发点
        obs["has_goal"] = self.has_goal
        self.last_pos = self.agent_pos
        # print("after reward truncated, info, done", reward, truncated, info, done)
        return obs, reward, done, truncated, info

    def reset(self,
              *,
              seed=None,
              options=None):
        self.has_goal = False
        obs, info = super().reset(seed=seed)
        obs["has_goal"] = self.has_goal
        self.last_pos = self.agent_pos
        return obs, info


def play_fetch_return():
    # 使用自定义环境
    env = FetchReturnEnv(size=10, render_mode="rgb_array", gen_obstacle=False)
    policy = BasePolicy()
    # 重置环境
    obs, info = env.reset()
    frames = []
    # 渲染初始状态
    frame = env.render()
    frames.append(frame)
    plt.imshow(frame)
    plt.show()

    # 示例：执行一些随机动作
    for _ in range(10):
        # action = env.action_space.sample()
        action = policy.get_action(obs)
        print("dir", obs["direction"], obs["has_goal"])
        obs, reward, done, truncated, info = env.step(action)
        frame = env.render()
        if reward > 0:
            pixel = 100 + int(reward * 150)
            print(pixel)
            frame[0:50, 0:50, 0] = pixel
        frames.append(frame)
        plt.imshow(frame)
        plt.show()
        if done:
            obs, info = env.reset()
            frame = env.render()
            frames.append(frame)
            plt.imshow(frame)
            plt.show()

    env.close()
    imageio.mimsave(f'./video/example_fetch_return.gif', frames, fps=8)


if __name__ == '__main__':
    # Simple_DoorKey()
    play_fetch_return()

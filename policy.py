import torch


class SimpleNN:
    def __init__(self, w, h, n):
        self.obs_w = w
        self.obs_h = h
        self.action_num = n

        self.emb_layer = torch.nn.Sequential(
            torch.nn.Linear(self.obs_h * self.obs_w + 2, 1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 256),
            torch.nn.ReLU(),
        )
        self.action_layer = torch.nn.Sequential(
            torch.nn.Linear(256, 7),
            torch.nn.Softmax()
        )#torch.nn.Linear(256,7)
        self.critic_layer = torch.nn.Linear(256,1)

    def forward(self, obs: torch.Tensor, last_a, last_r):
        b,w,h = obs.shape
        obs = obs.reshape(b, w*h)
        state = torch.cat([obs, last_a, last_r], dim=-1)
        print("check state shape", state)
        s = self.emb_layer(state)
        logit = self.action_layer(s)
        v = self.critic_layer(s)
        return logit, v


class BasePolicy:
    def __init__(self):
        pass

    def get_action(self, obs):
        """
        Num Name  Action
        0  left  Turn left
         1  right  Turn right
         2  forward  Move forward
        3  pickup  Pick up an object
        4  dropUnused
         5  toggle  Toggle/activate an object
        6  done  Unused
        :param obs:
        :return:
        """
        k = input("get new action")
        action_dict = {"a":0, "d":1, "w":2, "s":3, "x":4, "z":5, "c":6}
        if k in action_dict.keys():
            return action_dict[k]
        print("your input is no use", action_dict)
        return 6


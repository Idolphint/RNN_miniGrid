import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

device = torch.device("cuda:0")

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.hidden = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.hidden[:]


class RNN_AC(nn.Module):
    def __init__(self, obs_dim, state_dim=64, action_dim=7):
        super(RNN_AC, self).__init__()
        self.obs_dim = obs_dim
        self.state_dim = state_dim  # state_dim=hidden_dim=out_dim

        self.initializer = nn.Linear(obs_dim, state_dim)

        self.pre_embedding = nn.Sequential(
            nn.Linear(obs_dim, state_dim),
            nn.LeakyReLU(),
        )

        self.RNN = nn.GRU(state_dim, state_dim, num_layers=1)

        # self.RNN_norm = nn.LayerNorm(state_dim)

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def init_hidden(self, batch_size=1):
        return torch.zeros((batch_size, self.state_dim)).to(device)
        # return torch.normal(mean=torch.ones((batch_size, self.state_dim))).to(device)

    # def init_hidden(self, init_input, set_cuda=False):
    #     if set_cuda:
    #         init_input = torch.FloatTensor(init_input).to(device)
    #     return torch.tanh(self.initializer(init_input)).to(device)

    def emb(self, obs, dones, hidden):  # TODO 不是很能确定这个done重置的正不正确
        """
        :param obs: obs shape: [seq,batch,feature]或者[seq,feature]
        :param  dones: [seq, 1]
        :param  hidden: shape [1*num_layer, hidden] or [1*num_layer, batch, hidden]  # 本应输入的hidden
        :return: output: [seq, 1*hidden] or [seq, batch, 1*hidden]
                 h_n: after_hidden state: [num_layer, hidden] or 中间多一batch维度
        """
        # obs shape: [seq,batch,feature]或者[seq,feature]
        rnn_in = self.pre_embedding(obs)
        # 根据是否done选择重置hidden
        if len(obs.shape) == 3:
            batch_size = obs.shape[1]
        else:
            batch_size = 1
        # hidden要么是在交互中及时计算的，要么是存储好的，所以不用这里处理, 额外的问题，是否保留梯度，应当不保留了
        # print("check dones shape", dones.shape, obs.shape, hidden.shape)
        # hidden = torch.where(dones, self.init_hidden(obs), hidden)
        embedding, hidden = self.RNN(rnn_in, hidden)
        # hidden = self.RNN_norm(hidden)
        # print("see norm embedding and ori hidden", embedding, hidden)
        return hidden, embedding

    def act(self, x):  # TODO
        obs, dones, hidden = x
        hidden, embedding = self.emb(obs, dones, hidden)
        action_probs = self.actor(embedding)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(embedding)

        return action.detach(), action_logprob.detach(), state_val.detach(), hidden

    def evaluate(self, x, action):
        # 需要注意，这里将整个流程当作multi batch而不是seq数据
        obs, dones, hidden = x
        obs = torch.unsqueeze(obs, dim=0)
        hidden = torch.unsqueeze(hidden, dim=0)
        dones = torch.unsqueeze(dones, dim=0).unsqueeze(dim=2)
        hidden, embedding = self.emb(obs, dones, hidden)
        action_probs = self.actor(embedding)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(embedding)

        return action_logprobs, state_values, dist_entropy


class PPO_rnn:
    def __init__(self, state_dim, action_dim, ac_name="RNN",
                 lr_actor=1e-4, lr_critic=1e-4, gamma=0.9, K_epochs=10, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.ac_name = ac_name

        self.buffer = RolloutBuffer()
        if ac_name == "RNN":
            self.policy = RNN_AC(obs_dim=state_dim, action_dim=action_dim).to(device)
            self.optimizer = torch.optim.Adam([
                {'params': self.policy.pre_embedding.parameters(), 'lr': lr_actor},
                {'params': self.policy.RNN.parameters(), 'lr': lr_actor},
                {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                {'params': self.policy.critic.parameters(), 'lr': lr_critic}
            ])

            self.policy_old = RNN_AC(obs_dim=state_dim, action_dim=action_dim).to(device)
            self.policy_old.load_state_dict(self.policy.state_dict())
        # else:  # default
        #     self.policy = ActorCritic(state_dim, action_dim).to(device)
        #     self.optimizer = torch.optim.Adam([
        #         {'params': self.policy.actor.parameters(), 'lr': lr_actor},
        #         {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        #     ])
        #
        #     self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        #     self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, hidden):
        # with torch.no_grad():
        self.buffer.hidden.append(hidden)

        if len(self.buffer.actions) > 0:
            state[49, 1] = self.buffer.actions[-1].item()
            state[49, 2] = self.buffer.rewards[-1]
            dones = [[self.buffer.is_terminals[-1]]]
        else:
            dones = [[1]]
        state = state.reshape(1, -1)

        # print(state[0,100:])
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            dones = torch.BoolTensor(dones).to(device)
            if self.ac_name == "RNN":
                x = (state, dones, hidden)
            else:
                x = state
            action, action_logprob, state_val, hidden = self.policy_old.act(x)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.item(), hidden

    def update(self):
        print("before update see actions", self.buffer.actions)
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)
        old_hidden = torch.squeeze(torch.stack(self.buffer.hidden, dim=0)).detach().to(device)
        before_done = [1] + self.buffer.is_terminals[:-1]
        old_dones = torch.squeeze(torch.BoolTensor(before_done)).detach().to(device)

        if self.ac_name == "RNN":
            x = (old_states, old_dones, old_hidden)
        else:
            x = old_states
        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()
        # 需要注意，在update里，输入的state如何结合old_action, old_reward, old_done？
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(x, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

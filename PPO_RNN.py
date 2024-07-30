import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from rolloutBuffer import RolloutBuffer
from torch.distributions import Categorical
# from mamba_ssm import Mamba

device = torch.device("cuda:0")


class RNN_AC(nn.Module):
    def __init__(self, obs_dim, hidden_dim=128, action_dim=7):
        super(RNN_AC, self).__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim  # state_dim=hidden_dim=out_dim

        self.initializer = nn.Linear(obs_dim, hidden_dim)

        self.pre_embedding = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LeakyReLU(),
        )

        self.RNN = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)

        # self.RNN_norm = nn.LayerNorm( hidden_dim)

        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def init_hidden(self, batch_size=1):
        return torch.zeros((batch_size, self.hidden_dim)).to(device)
        # return torch.normal(mean=torch.ones((batch_size, self. hidden_dim))).to(device)

    # def init_hidden(self, init_input, set_cuda=False):
    #     if set_cuda:
    #         init_input = torch.FloatTensor(init_input).to(device)
    #     return torch.tanh(self.initializer(init_input)).to(device)

    def emb(self, obs, hidden):  # TODO 不是很能确定这个done重置的正不正确
        """
        :param obs: obs shape: [seq,batch,feature]或者[seq,feature]
        param  dones: [seq, 1]
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
        obs, hidden = x
        hidden, embedding = self.emb(obs, hidden)
        action_probs = self.actor(embedding)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(embedding)

        return action.detach(), action_logprob.detach(), state_val.detach(), action_probs.detach(), hidden

    def evaluate(self, x, action):
        #
        obs, hidden = x
        # obs = torch.unsqueeze(obs, dim=0)
        hidden = torch.unsqueeze(hidden, dim=0)
        # dones = torch.unsqueeze(dones, dim=0).unsqueeze(dim=2)
        hidden, embedding = self.emb(obs, hidden)

        embedding = embedding.reshape(embedding.shape[0]*embedding.shape[1], embedding.shape[2])

        action_probs = self.actor(embedding)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(embedding)

        return action_logprobs, state_values, dist_entropy, action_probs


class PPO_rnn:
    def __init__(self, state_dim, action_dim, hidden_dim=128, ac_name="RNN",
                 lr_actor=1e-4, lr_critic=1e-4, gamma=0.9, K_epochs=10, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.ac_name = ac_name
        self.global_update_step = 0
        self.hidden_dim = hidden_dim

        self.buffer = RolloutBuffer(obs_size=state_dim, hidden_size=hidden_dim)
        if ac_name == "RNN":
            self.policy = RNN_AC(obs_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim).to(device)
            self.optimizer = torch.optim.Adam([
                {'params': self.policy.pre_embedding.parameters(), 'lr': lr_actor},
                {'params': self.policy.RNN.parameters(), 'lr': lr_actor},
                {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                {'params': self.policy.critic.parameters(), 'lr': lr_critic}
            ])
        self.entropy_coef = 0.01  # 可以存储进state_dict从中恢复
        self.entropy_coef_step = 0.01 / 100000

        self.MseLoss = nn.MSELoss()
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return torch.zeros((1, 1, self.hidden_dim)).to(device)

    def play(self, env, state, test=True):
        if test:
            self.policy.eval()
        last_a, last_r, last_d = self.buffer.get_last_action_reward_done()
        state[-8+last_a] = 1
        state[-1] = last_r
        # 在play的时候也准备好了 batch,seq维度
        tensor_state = torch.FloatTensor(state.reshape(1, 1, -1)).to(device)
        x = (tensor_state, self.hidden)
        action, action_logprob, state_val, policy, new_hidden = self.policy.act(x)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated:
            if self.buffer.step_count < self.buffer.max_eps_length:
                self.buffer.add_data(
                    state=torch.from_numpy(state),
                    hidden=self.hidden.squeeze(),
                    action=action,
                    value=state_val.item(),
                    reward=reward,
                    done=1,
                    prob=action_logprob,
                    policy=policy
                )
            self.buffer.batch["dones_indices"][self.buffer.game_count] = self.buffer.step_count
            self.buffer.game_count += 1
            self.buffer.step_count = 0
        else:
            if self.buffer.step_count < self.buffer.max_eps_length:
                self.buffer.add_data(
                    state=torch.from_numpy(state),
                    hidden=self.hidden.squeeze(),
                    action=action,
                    value=state_val.item(),
                    reward=reward,
                    done=0,
                    prob=action_logprob,
                    policy=policy
                )
            self.buffer.step_count += 1
        # if store hidden here then update it
        self.hidden = new_hidden
        if terminated or truncated:
            self.hidden = self.init_hidden()

        return action, observation, reward, terminated, truncated

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
                x = (state, hidden)
            else:
                x = state
            action, action_logprob, state_val, hidden = self.policy_old.act(x)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.item(), hidden

    def _entropy_coef_schedule(self):
        self.entropy_coef -= self.entropy_coef_step
        if self.entropy_coef <= 0:
            self.entropy_coef = 0

    def update_new(self, writer):
        self.policy.train()
        self.policy.RNN.train()
        self.buffer.cal_advantages(self.gamma, gae_lambda=0.95)
        self.buffer.prepare_batch()

        for _ in range(self.K_epochs):
            mini_batch_generator = self.buffer.mini_batch_generator()

            for mini_batch in mini_batch_generator:
                for key, v in mini_batch.items():
                    if key == "values" or key == "probs" or key == "advantages" or key == "hidden":
                        mini_batch[key] = v.detach().to(device)
                    else:
                        mini_batch[key] = v.to(device)
                B, S = mini_batch["states"].shape
                # TODO 这里似乎是batch first？ 后面要改过来吗？
                mini_batch["states"] = mini_batch["states"].view(B // self.buffer.actual_sequence_length,
                                                                 self.buffer.actual_sequence_length, S)
                x = (mini_batch["states"], mini_batch["hidden"])
                # action, action_logprob, val_new, pol_new, _ = self.policy.act(x)
                log_prob_new, val_new, entropy, pol_new = self.policy.evaluate(x, mini_batch["actions"].view(1, -1))
                # pol_new, val_new, _, _ = self.policy(mini_batch["states"], mini_batch["h_states"].unsqueeze(0),
                #                                     mini_batch["c_states"].unsqueeze(0))
                val_new = val_new.squeeze(1)
                # log_prob_new, entropy = self.distribution.log_prob(pol_new, mini_batch["actions"].view(1, -1),
                #                                                    mini_batch["action_mask"])
                log_prob_new = log_prob_new.squeeze(0)[mini_batch["loss_mask"]]
                val_new = val_new[mini_batch["loss_mask"]]
                entropy = entropy[mini_batch["loss_mask"]]

                ratios = torch.exp(log_prob_new - mini_batch["probs"].reshape(-1).detach())

                # Finding Surrogate Loss
                surr1 = ratios * mini_batch["advantages"]
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mini_batch["advantages"]

                # final loss of clipped objective PPO
                actor_loss = -torch.min(surr1, surr2)
                critic_loss = self.MseLoss(mini_batch["values"] + mini_batch["advantages"], val_new)

                writer.add_scalar("Loss/ActorLoss", actor_loss.mean(), global_step=self.global_update_step)
                writer.add_scalar("Loss/CriticLoss", critic_loss.mean(), global_step=self.global_update_step)
                writer.add_scalar("Loss/Entropy", entropy.mean(), global_step=self.global_update_step)
                self.global_update_step += 1
                total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy
                with torch.autograd.set_detect_anomaly(True):
                    if not torch.isnan(total_loss).any():
                        self.optimizer.zero_grad()
                        total_loss.mean().backward()
                        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                        self.optimizer.step()
                        self._entropy_coef_schedule()  # slow down entropy coef

            self.buffer.reset_data()

    def update(self, writer):
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
            actor_loss = -torch.min(surr1, surr2)
            critic_loss = self.MseLoss(state_values, rewards)

            writer.add_scalar("Loss/ActorLoss", actor_loss.mean(), global_step=self.global_update_step)
            writer.add_scalar("Loss/CriticLoss", critic_loss.mean(), global_step=self.global_update_step)
            writer.add_scalar("Loss/Entropy", dist_entropy.mean(), global_step=self.global_update_step)
            self.global_update_step += 1

            loss = actor_loss + 0.5 * critic_loss - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        # self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

import math
import torch
from time import time
import numpy as np
from torch.nn import MSELoss
import torch.distributions as dist

 

class SAC:
    # recurrent soft actor critic
    def __init__(
            self,
            policy_net,
            soft_q_net_1, soft_q_net_2,
            target_q_net_1, target_q_net_2,
            q_loss, # NStepQValueLossSeparateEntropy.
            policy_optimizer, q_optim_1, q_optim_2,
            soft_tau, device,
            action_dim, 
            n_steps=10, # 10.
            eta=0.9,
            q_dim=1, q_weights=None,
            use_observation_normalization=False
    ):
        self.soft_tau = soft_tau
        self.device = device

        self.policy_net = policy_net
        self.soft_q_net_1 = soft_q_net_1
        self.soft_q_net_2 = soft_q_net_2
        self.target_q_net_1 = target_q_net_1
        self.target_q_net_2 = target_q_net_2
        self.q_loss = q_loss

        self.policy_optimizer = policy_optimizer
        self.q_optim_1 = q_optim_1
        self.q_optim_2 = q_optim_2

        self.target_entropy = -action_dim # -22.
        self.sac_log_alpha = torch.tensor(
            0,
            dtype=torch.float32,
            requires_grad=True,
            device=device
        )
        self.sac_alpha = self.sac_log_alpha.exp().item()
        self.alpha_optim = torch.optim.Adam([self.sac_log_alpha], lr=1e-3)

        self.n_steps = n_steps # 10.
        self.eta = eta  # priority weight

        if q_weights is None:
            q_weights = [1.0 for _ in range(q_dim)]
        self.q_dim = q_dim
        self.q_weights = torch.tensor(
            [[q_weights]],
            dtype=torch.float32,
            device=self.device
        )
        if use_observation_normalization:
            self.norm_obs = True
            mean_and_std = np.load('obs_mean_and_std.npy')
            obs_mean, obs_std = np.split(mean_and_std, 2, axis=0)
            obs_mean = np.concatenate(
                [np.zeros(11 * 11 * 2, dtype=float), obs_mean[0], [0.0]], axis=0
            )
            obs_std = np.concatenate(
                [np.ones(11 * 11 * 2, dtype=float), obs_std[0], [1.0]], axis=0
            )
            self.obs_mean = torch.tensor(
                [[obs_mean]], dtype=torch.float32, device=self.device
            )
            self.obs_std = torch.tensor(
                [[obs_std]], dtype=torch.float32, device=self.device
            )
        else:
            self.norm_obs = False

    def save(self, filename):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'log_alpha': self.sac_log_alpha.item(),
            'q_net_1': self.soft_q_net_1.state_dict(),
            'q_net_2': self.soft_q_net_2.state_dict(),

            'policy_optim': self.policy_optimizer.state_dict(),
            'q_net_1_optim': self.q_optim_1.state_dict(),
            'q_net_2_optim': self.q_optim_2.state_dict(),
            'alpha_optim': self.alpha_optim.state_dict()
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.sac_log_alpha.data = torch.tensor(
            checkpoint['log_alpha'],
            dtype=torch.float32,
            device=self.device
        )
        self.sac_alpha = self.sac_log_alpha.exp().item()
        self.soft_q_net_1.load_state_dict(checkpoint['q_net_1'])
        self.soft_q_net_2.load_state_dict(checkpoint['q_net_2'])
        self.target_q_net_1.load_state_dict(checkpoint['q_net_1'])
        self.target_q_net_2.load_state_dict(checkpoint['q_net_2'])

        if 'policy_optim' in checkpoint:
            self.policy_optimizer.load_state_dict(checkpoint['policy_optim'])
            self.q_optim_1.load_state_dict(checkpoint['q_net_1_optim'])
            self.q_optim_2.load_state_dict(checkpoint['q_net_2_optim'])
        if 'alpha_optim' in checkpoint:
            self.alpha_optim.load_state_dict(checkpoint['alpha_optim'])



    def load_policy(self, filename):
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.sac_log_alpha.data = torch.tensor(
            checkpoint['log_alpha'],
            dtype=torch.float32,
            device=self.device
        )



    def compute_mask(self, is_done): # is_done: (n_env, T).
        '''        
        if first done happend at t <= T-1, then mask[t+1:T] will all be 0.
        '''

        mask = torch.ones_like(is_done)
        mask[:, 1:] = 1.0 - (is_done[:, :-1].cumsum(-1) > 0).to(torch.float32)
 
        return mask[:, -self.n_steps:]
    


    def train(self):
        self.policy_net.train()
        self.soft_q_net_1.train()
        self.soft_q_net_2.train()


    def eval(self):
        self.policy_net.eval()
        self.soft_q_net_1.eval()
        self.soft_q_net_2.eval()


    def actor_train(self):
        self.policy_net.train()



    def actor_eval(self):
        self.policy_net.eval()



    def state_to_numpy(self, actor_state, q_1_state, q_2_state):
        numpy_actor_state = self.policy_net.state_to_numpy(actor_state)
        numpy_critic_state_1 = self.soft_q_net_1.state_to_numpy(q_1_state)
        numpy_critic_state_2 = self.soft_q_net_2.state_to_numpy(q_2_state)
        numpy_critic_state = np.concatenate(
            (numpy_critic_state_1, numpy_critic_state_2),
            axis=-1
        )
        return numpy_actor_state, numpy_critic_state


    def state_from_numpy(self, actor_state, critic_state):
        actor_state_t = self.policy_net.state_from_numpy(actor_state, self.device)
        q_1_state, q_2_state = np.split(critic_state, 2, axis=-1)
        q_1_state_t = self.soft_q_net_1.state_from_numpy(q_1_state, self.device)
        q_2_state_t = self.soft_q_net_2.state_from_numpy(q_2_state, self.device)
        return actor_state_t, q_1_state_t, q_2_state_t



    def zero_state(self, actor_state, critic_state, done):

        done = torch.tensor(done, dtype=torch.float32, device=self.device)

        actor_state, q_1_state, q_2_state = self.state_from_numpy(
            actor_state, critic_state
        )

        actor_state = self.policy_net.zero_state(actor_state, done)
        q_1_state = self.soft_q_net_1.zero_state(q_1_state, done)
        q_2_state = self.soft_q_net_2.zero_state(q_2_state, done)

        actor_state, critic_state = self.state_to_numpy(
            actor_state, q_1_state, q_2_state
        )
        return actor_state, critic_state




    def sample_action_log_prob(self, observation_t, policy_state_t):

        '''
        observation_t: (b, T+1, dim), T=10.
        '''

        # mean, log_std: (b, T+1, action_dim).
        mean, log_std, _ = self.policy_net(observation_t, policy_state_t)

        std = log_std.exp()
        distribution = dist.Normal(mean, std)
        z = distribution.rsample() # (b, T+1, action_dim).
        action = torch.tanh(z) # (b, T+1, action_dim).
        log_prob = distribution.log_prob(z) # (b, T+1, action_dim).

        # calculate logarithms like a noob:
        # log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6)

        # calculate logarithms like a pro:
        log_prob = log_prob - math.log(4.0) + 2 * torch.log(z.exp() + (-z).exp()) # (b, T+1, action_dim).

        log_prob = log_prob.sum(-1) # (b, T+1).
        return action, log_prob  # action: (b, T+1, action_dim), # log_prob: (b, T+1).
    


    def act_test(self, observation, policy_state=None):
        observation_t = torch.tensor(
            [[observation]],  # add batch and time dimensions
            dtype=torch.float32,
            device=self.device
        )
        if self.norm_obs:
            observation_t = (observation_t - self.obs_mean) / self.obs_std
        if policy_state is not None:
            policy_state = self.policy_net.state_from_numpy(
                policy_state, self.device
            )
        action, policy_state = self.act(observation_t, policy_state)
        action = action[0, 0].cpu().numpy()  # select batch and time
        policy_state = self.policy_net.state_to_numpy(policy_state)
        return action, policy_state



    def act(self, observation_t, policy_state_t=None):
        with torch.no_grad():
            mean, log_std, new_state = self.policy_net(observation_t, policy_state_t)
        if self.policy_net.training:
            batch_size = mean.size(0)
            std = log_std.exp()
            distribution = dist.Normal(mean, std)
            action_t = torch.tanh(mean)
            action_t[batch_size // 2:] = torch.tanh(distribution.sample()[batch_size // 2:])
        else:
            action_t = torch.tanh(mean)
        return action_t, new_state



    def act_q(self, observation,
              policy_state=None, q_state=None):
        
        observation_t = torch.tensor(
            observation,
            dtype=torch.float32,
            device=self.device
        )

        observation_t.unsqueeze_(1)

        if self.norm_obs:
            observation_t = (observation_t - self.obs_mean) / self.obs_std

        torch.Size()

        if policy_state is not None and q_state is not None:
            policy_state, q_1_state, q_2_state = self.state_from_numpy(policy_state, q_state)
        else:
            q_1_state, q_2_state = None, None

        action_t, new_policy_state = self.act(observation_t, policy_state)
        action = action_t.squeeze(1)
        action = action.cpu().numpy()

        with torch.no_grad():
            _, new_q_1_state = self.soft_q_net_1(observation_t, action_t, q_1_state)
            _, new_q_2_state = self.soft_q_net_2(observation_t, action_t, q_2_state)

        new_policy_state, new_q_state = self.state_to_numpy(
            new_policy_state, new_q_1_state, new_q_2_state
        )

        return action, new_policy_state, new_q_state



    def batch_to_tensors(self, batch):
        def t(x):
            return torch.tensor(x, dtype=torch.float32, device=self.device)

        observation_t, actions, rewards, is_done = map(t, batch)

        if self.norm_obs:
            observation_t = (observation_t - self.obs_mean) / self.obs_std

        return observation_t, actions, rewards, is_done



    def calculate_priority(self, q_1_loss, q_2_loss, segment_length):
        '''
        q_1_loss, q_2_loss: (n_env, T). segment_length: (n_env,).
        '''
         
        q_loss = torch.sqrt(2.0 * torch.max(q_1_loss, q_2_loss)) # (n_env, T).
        max_over_time = torch.max(q_loss, dim=1)[0] # (,)?! "[0]" should be removed?
        mean_over_time = q_loss.sum(dim=1) / segment_length # (n_env,).

        priority_loss = self.eta * max_over_time + (1 - self.eta) * mean_over_time # (n_env,).
        return (priority_loss.detach() + 1e-6).cpu().numpy() # (n_env,).



    def calculate_priority_loss(self, data, policy_state, q_state):

        '''
            data:
                obs: (n_env, T+1, dim). act, rew: (n_env, T, dim). don: (n_env, T). T==10.
        '''

        # almost same as q_value_loss
        observations, actions, rewards, is_done = self.batch_to_tensors(data)

        # if first done for env i happend at t <= T-1, then mask[i, t+1:T] will all be 0.
        mask = self.compute_mask(is_done) # (n_env, T=10)
        
        segment_length = mask.sum(-1) + 1 # (n_env,)
        policy_state, q_1_state, q_2_state = self.state_from_numpy(
            policy_state, q_state
        )

        with torch.no_grad():
            q_1_loss, q_2_loss = self.calc_q_value_loss( # (n_env, T), (n_env, T)
                observations, actions, rewards, is_done, mask,
                policy_state, q_1_state, q_2_state,
            )

        
        priority_loss = self.calculate_priority(q_1_loss, q_2_loss, segment_length)  # (n_env,).
        return priority_loss # (n_env,).



    def calc_q_value_loss(self, observations, actions, rewards, is_done, mask,
                          policy_state, q_1_state, q_2_state):
        '''
        - T==10.
        - observations: (n_env, T+1, dim). actions, rewards: (n_env, T, dim). is_done: (n_env, T). 
        - mask: (n_env, T).
        '''

        current_q_1, _ = self.soft_q_net_1(observations[:, :-1], actions, q_1_state) # (n_env, T, dim).
        current_q_2, _ = self.soft_q_net_2(observations[:, :-1], actions, q_2_state) # (n_env, T, dim).

        with torch.no_grad():
            # action: (n_env, T+1, action_dim).
            action, log_prob = self.sample_action_log_prob(observations, policy_state)
            next_state_q_1, _ = self.target_q_net_1(observations, action, q_1_state) # (n_env, T+1, dim).
            next_state_q_2, _ = self.target_q_net_2(observations, action, q_2_state)


        min_q = torch.min(next_state_q_1[:, 1:], next_state_q_2[:, 1:]) # (n_env, T, dim).
        
        # next_state_value = min_q - self.sac_alpha * log_prob[:, 1:]  # [B, T, q_dim]
        next_state_value = min_q # (n_env, T, dim).
        log_p_for_loss = -self.sac_alpha * log_prob[:, 1:] # (n_env, T, dim).

        # roi
        # n_steps: 10 == T.
        current_q_1 = current_q_1[:, -self.n_steps:] # no effect.
        current_q_2 = current_q_2[:, -self.n_steps:] # no effect.
        next_state_value = next_state_value[:, -self.n_steps:] # no effect.
        log_p_for_loss = log_p_for_loss[:, -self.n_steps:]
        rewards = rewards[:, -self.n_steps:] # no effect.
        is_done = is_done[:, -self.n_steps:] # no effect.


        # loss

        q_1_loss = self.q_loss( # NStepQValueLossSeparateEntropy. # (n_env, T, dim). 
            current_q_1, next_state_value, log_p_for_loss, rewards, is_done, mask
        )
        q_2_loss = self.q_loss( # NStepQValueLossSeparateEntropy. # (n_env, T, dim). 
            current_q_2, next_state_value, log_p_for_loss, rewards, is_done, mask
        )

        q_1_loss = q_1_loss.sum(-1) # (n_env, T). 
        q_2_loss = q_2_loss.sum(-1) # (n_env, T). 
        return q_1_loss, q_2_loss



    def soft_target_update(self):
        for p, tp in zip(self.soft_q_net_1.parameters(), self.target_q_net_1.parameters()):
            tp.data.copy_(
                (1.0 - self.soft_tau) * tp.data + self.soft_tau * p.data
            )
        for p, tp in zip(self.soft_q_net_2.parameters(), self.target_q_net_2.parameters()):
            tp.data.copy_(
                (1.0 - self.soft_tau) * tp.data + self.soft_tau * p.data
            )


    def optimize_q(self, q_1_loss, q_2_loss, importance_weights, segment_len):
        '''
        q_1_loss, q_2_loss: (n_env, T). 
        segment_len: (n_env,)
        '''
        self.q_optim_1.zero_grad()
        q_1_loss = (importance_weights * q_1_loss.sum(-1) / segment_len).mean()
        q_1_loss.backward()
        self.q_optim_1.step()

        self.q_optim_2.zero_grad()
        q_2_loss = (importance_weights * q_2_loss.sum(-1) / segment_len).mean()
        q_2_loss.backward()
        self.q_optim_2.step()

        return q_1_loss.item(), q_2_loss.item()



    def optimize_p(self, policy_loss, alpha_loss, importance_weights, segment_len):

        # policy_loss, alpha_loss: (b, T), segment_len: (b,).

        self.policy_optimizer.zero_grad()
        policy_loss = (importance_weights * policy_loss.sum(-1) / segment_len).mean() # (,)
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1)
        policy_loss.backward()
        self.policy_optimizer.step()

        self.alpha_optim.zero_grad()
        alpha_loss = (importance_weights * alpha_loss.sum(-1) / segment_len).mean()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.sac_alpha = self.sac_log_alpha.exp().item()

        return policy_loss.item(), alpha_loss.item()



    def learn_q_from_data(self,
                          importance_weights,
                          observations, actions, rewards, is_done,
                          mask, 
                          segment_length, # (b,)
                          policy_state, q_1_state, q_2_state):
        



        q_1_loss, q_2_loss = self.calc_q_value_loss( # (n_env, T). 
            observations, actions, rewards, is_done, mask,
            policy_state, q_1_state, q_2_state
        )

        
        priority = self.calculate_priority(q_1_loss, q_2_loss, segment_length) # (n_env,).

        q_1_loss, q_2_loss = self.optimize_q( # (,), (,)
            q_1_loss, q_2_loss, importance_weights, segment_length
        )

        
        return q_1_loss, q_2_loss, priority





    def calc_policy_loss(self, observations, 
                         mask, # (b, T),
                         policy_state, q_1_state, q_2_state):
        
        # forward, log_pi

        # observations: (b, T+1, dim). 
        # action: (b, T+1, action_dim), log_prob: (b, T+1).
        actions, log_prob = self.sample_action_log_prob(observations, policy_state) # uses policy net.

        # target log_prob, Q
        q_1, _ = self.soft_q_net_1(observations, actions, q_1_state) # (b, T+1, q_dim).
        q_2, _ = self.soft_q_net_2(observations, actions, q_2_state) # (b, T+1, q_dim).

        q_min = torch.min(q_1, q_2) # (b, T+1, q_dim). 
        target_log_prob = (self.q_weights * q_min).sum(-1) # (b, T+1).

        # roi
        log_prob = log_prob[:, -self.n_steps:] # (b, T).
        target_log_prob = target_log_prob[:, -self.n_steps:] # (b, T).

        # policy and alpha losses
        policy_loss = mask * (self.sac_alpha * log_prob - target_log_prob) # (b, T),
        alpha_loss = -(self.sac_log_alpha * (log_prob + self.target_entropy).detach()) # (b, T).
        alpha_loss = mask * alpha_loss  # (b, T).

        # std = mask * log_std[:, -(self.n_steps + 1):-1].exp().mean(-1)  # [B, T + 1, action_dim]
        return policy_loss, alpha_loss, q_min
    
        



    def learn_p_from_data(self,
                          importance_weights,
                          observations, mask, segment_length,
                          policy_state, q_1_state, q_2_state):
        '''
            observations: (b, T+1, dim).  
            segment_length: (b,), 
            mask: (b, T),
            T=10.      
        '''
                
        # policy_loss, alpha_loss: (b, T), q_min: (b, T+1, q_dim).                
        policy_loss, alpha_loss, q_min = self.calc_policy_loss(
            observations, mask,
            policy_state, q_1_state, q_2_state
        )

        
        policy_loss, alpha_loss = self.optimize_p( # (,)
            policy_loss, alpha_loss, importance_weights, segment_length
        )

        # std = (std.sum(-1) / segment_length).mean().item()
        mean_q_min = (q_min.sum(1) / segment_length.unsqueeze(-1)).mean(0)  # (q_dim,)
        mean_q_min = mean_q_min.detach().cpu().numpy()
        return policy_loss, alpha_loss, mean_q_min # (,), (,), (q_dim,)




    def learn_from_data(self, data, importance_weights=1.0,  # multiply gradients by 1 is always ok
                        policy_state=None, q_state=(None, None),
                        learn_policy=True  # True.
                        ):
     
        # observations: (b, T+1, dim). actions, rewards: (b, T, dim). is_done: (b, T).  
        observations, actions, rewards, is_done = self.batch_to_tensors(data)
        mask = self.compute_mask(is_done) # (b, T=10)


        # добавить везде единицу в segment_len - нормально,
        # потому что это уберет деление на ноль,
        # а градииенты на каждом шаге по времени изменятся одинаково
        segment_length = mask.sum(-1) + 1 # (b,)
        importance_weights = torch.tensor(
            importance_weights, dtype=torch.float32, device=self.device
        )

        policy_state, q_1_state, q_2_state = self.state_from_numpy(policy_state, q_state)


        # q_1_loss, q_2_loss: (,). priority: (b,).
        q_1_loss, q_2_loss, priority = self.learn_q_from_data(
            importance_weights,
            observations, actions, rewards, is_done,
            mask, segment_length,
            policy_state, q_1_state, q_2_state
        )
        self.soft_target_update()


        if learn_policy: # yes.
            policy_loss, alpha_loss, q_min = self.learn_p_from_data( # (,), (,), (q_dim,)
                        importance_weights, observations, mask, segment_length,
                        policy_state, q_1_state, q_2_state)
            
        else:
            policy_loss, alpha_loss = [0 for _ in range(2)]
            q_min = [0 for _ in range(self.q_dim)]


        mean_batch_reward = (self.q_weights * rewards[:, -self.n_steps:, :]).sum(-1)  # (b, T)
        mean_batch_reward = (mask * mean_batch_reward).sum(-1)  # (b,)
        mean_batch_reward = (mean_batch_reward / segment_length).mean().item() # (,)

        losses = np.array(
            [
                policy_loss, alpha_loss, # (,), (,)
                q_1_loss, q_2_loss, # (,), (,)
                mean_batch_reward # (,)
            ]
        ) # (5,)
 
        return losses, q_min, priority # (5,), (q_dim,), (b,)



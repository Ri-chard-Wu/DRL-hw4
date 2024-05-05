import torch
import torch.nn as nn


# DDPG
# may be used for RDPG as well
class PolicyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_values):
        return q_values.mean()


class QValueLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma
        # "reduction = 'none'" - for priority
        self.mse = nn.MSELoss(reduction='none') 

    def forward(self, current_q_value, next_q_value, log_p, reward, is_done, mask):
        current_q_value = current_q_value.squeeze(-1)
        next_q_value = next_q_value.squeeze(-1) + log_p
        target_q_value = reward + (1.0 - is_done) * self.gamma * next_q_value
        loss = self.mse(current_q_value, target_q_value.detach())
        return 0.5 * mask * loss


# value function rescaling https://arxiv.org/ abs/1805.11593
def id_fn(x):
    return x


def rescaling_fn(x):
    eps = 1e-3

    sign = torch.sign(x)
    sqrt = torch.sqrt(torch.abs(x) + 1)
    return sign * (sqrt - 1) + eps * x


def inv_rescaling_fn(x):
    eps = 1e-3

    sign = torch.sign(x)
    sqrt_arg = 1 + 4 * eps * (torch.abs(x) + 1 + eps)
    square = ((torch.sqrt(sqrt_arg) - 1) / (2 * eps)) ** 2
    return sign * (square - 1)




class NStepQValueLoss(nn.Module):
    # supports value function rescaling from 1805.11593
    def __init__(self, gamma, device, q_weights, n_steps=5, rescaling=False):
        super().__init__()
        self.gamma = gamma
        self.mse = nn.MSELoss(reduction='none')
        self.q_weights = torch.tensor(  # уже точно не помню для чего нужны ___в этой функции___
            [[q_weights]],
            dtype=torch.float32, device=device
        )
        # будем умножать лосс на эти веса, тогда из ненужных голов не будет учитываться градиент
        self.non_zero_q = (self.q_weights != 0).to(torch.float32)
        # специальные веса для того чтобы не добавлять энтропию в ненужные головы
        self.inv_q_weights = self._inv_q_weights(self.q_weights)
        # число ненулевых q-координат, нужно для правильного усреднения энтропии
        self.inv_num_q = 1.0 / sum([1 for qw in q_weights if qw != 0.0])
        self.n_steps = n_steps
        if rescaling:
            self.rescaling_fn = rescaling_fn
            self.inv_rescaling_fn = inv_rescaling_fn
        else:
            self.rescaling_fn = id_fn
            self.inv_rescaling_fn = id_fn
        self.gamma_t = torch.tensor(
            [[gamma ** i] for i in range(n_steps + 1)],
            dtype=torch.float32, device=device
        )

    @staticmethod
    def _inv_q_weights(q_weights):
        inv_q_weights = torch.zeros_like(q_weights)
        for i, q in enumerate(q_weights[0, 0]):
            if q != 0:
                inv_q = 1 / q
            else:
                inv_q = 0
            inv_q_weights[0, 0, i] = inv_q
        return inv_q_weights

    def forward(self,
                current_q_value, next_q_value,
                log_p, reward, is_done, mask):
        # current_q_values [B, T, q_dim]
        # next_q_values [B, T, q_dim]
        # log_p [B, T]
        # reward [B, T, q_dim]
        # is_done [B, T]
        # mask [B, T]
        batch, time, q_dim = reward.size()
        current_q_value = current_q_value

        log_p = log_p.unsqueeze(-1)
        mask = mask.unsqueeze(-1)
        is_done = is_done.unsqueeze(-1)

        next_q_value = mask * (1.0 - is_done) * self.inv_rescaling_fn(next_q_value)

        log_p = (mask * (1.0 - is_done) * log_p)
        log_p = self.inv_num_q * self.inv_q_weights * log_p
        log_p = log_p.expand(batch, time, q_dim)

        target_q_value = torch.zeros_like(reward)
        pad = torch.zeros(batch, self.n_steps - 1, q_dim, device=current_q_value.device)
        pad_reward = torch.cat((mask * reward, pad), dim=1)
        log_p_pad = torch.cat((log_p, pad), dim=1)

        for t in range(time):
            idx = min(time - 1, t + self.n_steps - 1)
            reward_sum = (pad_reward[:, t:t + self.n_steps, :] * self.gamma_t[:-1])  # aka 'return'
            log_p_sum = (log_p_pad[:, t:t + self.n_steps] * self.gamma_t[1:])
            next_q = (self.gamma ** idx) * next_q_value[:, idx]
            target_q_value[:, t] = reward_sum.sum(1) + next_q + log_p_sum.sum(1)
        target_q_value = self.rescaling_fn(target_q_value)
        loss = self.mse(current_q_value, target_q_value)
        loss = self.q_weights * loss
        return 0.5 * mask * loss










class NStepQValueLossSeparateEntropy(nn.Module):
    # include additional coordinate in Q-value for the entropy bonus
    # this class is practically same as the class above
    def __init__(self, gamma, device, q_weights, 
                 n_steps=5, # 5. 
                 rescaling=False # True.
                 ):
        super().__init__()
        self.gamma = gamma
        self.mse = nn.MSELoss(reduction='none') # just differenct & square, no mean.
        # expand q weights by one term for entropy. It is always 1.0 and modulated by sac_alpha
        self.q_weights = torch.tensor(
            [[q_weights + [1]]],
            dtype=torch.float32, device=device
        )
        self.non_zero_q = (self.q_weights != 0).to(torch.float32)
        self.n_steps = n_steps # 5.


        if rescaling: # yes.

            self.rescaling_fn = rescaling_fn
            self.inv_rescaling_fn = inv_rescaling_fn

        else:
            self.rescaling_fn = id_fn
            self.inv_rescaling_fn = id_fn



        self.gamma_t = torch.tensor( # discount factor
            [[gamma ** i] for i in range(n_steps + 1)],
            dtype=torch.float32, device=device
        ) # (n_steps + 1, 1)


    def forward(self,
                current_q_value, next_q_value,
                log_p, reward, is_done, mask):
        
        '''
            current_q_value: (n_env, T, dim). 
            next_q_value: (n_env, T, dim). 
            log_p: (n_env, T). 
            reward: (n_env, T, dim). 
            is_done: (n_env, T). 
            mask: (n_env, T). 
            T=10.
        '''

        # here current_q_value and next_q_value is tensors of size [B, T, q_dim + 1]
        # and reward is tensor of size [B, T, q_dim]
        B, T, q_dim = reward.size() # shape == (b, 10, q_dim)

        # mask: if first done for env i happend at t <= T-1, then mask[i, t+1:T] will all be 0.
        mask = mask.unsqueeze(-1) # (n_env, T, 1). 
        is_done = is_done.unsqueeze(-1) # (n_env, T, 1). 

        # "* (1.0 - is_done)" has no effect?
        next_q_value = mask * (1.0 - is_done) * self.inv_rescaling_fn(next_q_value)

        # [:q_dim] for reward, [q_dim:q_dim+1] for log prob.
        target_q_value = torch.zeros(B, T, q_dim + 1, device=current_q_value.device)

        # self.n_steps: 5, T: 10.
        pad = torch.zeros(B, self.n_steps - 1, q_dim, device=current_q_value.device)
        pad_reward = torch.cat((mask * reward, pad), dim=1)  # (n_env, T + self.n_steps - 1, q_dim)
         
        log_p_pad = torch.zeros((B, self.n_steps - 1), device=current_q_value.device)
        log_p_pad = torch.cat((log_p, log_p_pad), dim=1) # (n_env, T + self.n_steps - 1)


        for t in range(T): # T: 10.

            # idx: from 5 to 10 in first 5 iters, then stay 10 for last 5 iters.
            idx = min(T - 1, t + self.n_steps - 1) # self.n_steps: 5.

            # n-step return (not sumed yet).
            # gamma_t: discount factors, (n_steps + 1, 1).
            reward_to_sum = (pad_reward[:, t:t + self.n_steps, :] * self.gamma_t[:-1])  # aka 'return'

            # need to remove last dim of gamma_t for log_p_sum calculation
            log_p_to_sum = (log_p_pad[:, t:t + self.n_steps] * self.gamma_t[1:, 0])  # [B,]
            
            next_q = (self.gamma ** idx) * next_q_value[:, idx] # (n_env, dim). 

            target_q_value[:, t, :-1] = reward_to_sum.sum(1) # n-step return.
            target_q_value[:, t, -1] = log_p_to_sum.sum(1)
            target_q_value[:, t] = target_q_value[:, t] + next_q

        target_q_value = self.rescaling_fn(target_q_value)
        loss = self.mse(current_q_value, target_q_value) # (n_env, T, dim). 
        loss = self.q_weights * loss # (n_env, T, dim). 

        return 0.5 * mask * loss # (n_env, T, dim). 







class QuantileQValueLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ):
        pass

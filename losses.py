 
import tensorflow as tf 

 


# value function rescaling https://arxiv.org/ abs/1805.11593
  
# def rescaling_fn(x):
#     eps = 1e-3 
#     sign = tf.math.sign(x)
#     sqrt = tf.math.sqrt(tf.math.abs(x) + 1)
#     return sign * (sqrt - 1) + eps * x

 
# def inv_rescaling_fn(x): 
#     eps = 1e-3
#     sign = tf.math.sign(x)
#     sqrt_arg = 1 + 4 * eps * (tf.math.abs(x) + 1 + eps)
#     square = ((tf.math.sqrt(sqrt_arg) - 1) / (2 * eps)) ** 2
#     return sign * (square - 1)
 


  
def rescaling_fn(x):
    eps = 1e-3 
    sign = np.sign(x)
    sqrt = np.sqrt(np.abs(x) + 1)
    return sign * (sqrt - 1) + eps * x

 
def inv_rescaling_fn(x): 
    eps = 1e-3
    sign = np.sign(x)
    sqrt_arg = 1 + 4 * eps * (np.abs(x) + 1 + eps)
    square = ((np.sqrt(sqrt_arg) - 1) / (2 * eps)) ** 2
    return sign * (square - 1)
 





class NStepQValueLossSeparateEntropy():

    def __init__(self, gamma, q_weights, 
                 n_steps=5, # 5. 
                 rescaling=False # True.
                 ):

        super().__init__()
  
 
        self.gamma = gamma

        self.n_steps = n_steps # 5.

        self.gamma_t = tf.Variable(initial_value=[[gamma**i] for i in range(n_steps + 1)],
                                    trainable=False, dtype=tf.float32) # (n_steps + 1, 1)

    

    def compute_target_q(self,                 
                next_q, # (b, T, q_dim+1). 
                log_p, # (b, T). 
                reward,  # (b, T, q_dim). 
                is_done, # (b, T). 
                mask # (b, T).
            ):
        '''
        T == 10.
        '''   
        
        next_q = next_q.numpy() 
        log_p = log_p.numpy() 
        reward = reward.numpy() 
        is_done = is_done.numpy()[:,:,None]
        mask = mask.numpy()[:,:,None]

         
        B, T, q_dim = reward.shape # shape == (b, 10, q_dim)

        # mask: if first done for env i happend at t <= T-1, then mask[i, t+1:T] will all be 0.        

        # mask = tf.expand_dims(mask, axis=-1) # (b, T, 1). 
        # is_done = tf.expand_dims(is_done, axis=-1) # (b, T, 1).   

        # "* (1.0 - is_done)" has no effect?
        next_q = mask * (1.0 - is_done) * inv_rescaling_fn(next_q) # (b, T, q_dim+1). 
 
        # [:q_dim] for reward, [q_dim:q_dim+1] for log prob.
        target_q_value = np.zeros([B, T, q_dim + 1], dtype=np.float32)

        # self.n_steps: 5, T: 10.
        pad = np.zeros([B, self.n_steps - 1, q_dim], dtype=np.float32)
        # pad_reward = tf.concat([mask * reward, pad], axis=1) # (b, T + self.n_steps - 1, q_dim)
        pad_reward = np.concatenate([mask * reward, pad], axis=1)


        log_p_pad = np.zeros([B, self.n_steps - 1], dtype=np.float32)
        log_p_pad = np.concatenate([log_p, log_p_pad], axis=1) # (b, T + self.n_steps - 1)

  
        for t in range(T): # T: 10.
        
            idx = min(T - 1, t + self.n_steps - 1) # self.n_steps: 5.

            # n-step return (not sumed yet).
            # gamma_t: discount factors, (n_steps + 1, 1).
            reward_to_sum = (pad_reward[:, t:t + self.n_steps, :] * self.gamma_t[:-1]) # (b, n_steps, q_dim) 

            # need to remove last dim of gamma_t for log_p_sum calculation
            log_p_to_sum = (log_p_pad[:, t:t + self.n_steps] * self.gamma_t[1:, 0])  # (b,)
            
            next_q = (self.gamma ** idx) * next_q[:, idx] # (b, q_dim+1). 
 
            # n-step return.
            # target_q_value[:, t, :-1] = tf.reduce_sum(reward_to_sum, axis=1) # (b, q_dim)  
            target_q_value[:, t, :-1] = np.sum(reward_to_sum, axis=1)

            target_q_value[:, t, -1] = log_p_to_sum
            target_q_value[:, t] = target_q_value[:, t] + next_q  # (b, q_dim+1). 


        target_q_value = rescaling_fn(target_q_value) # (b, T, q_dim+1)

        return tf.convert_to_tensor(target_q_value, dtype=tf.float32)

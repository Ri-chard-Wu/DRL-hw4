import math
 
from time import time
import numpy as np

from losses import NStepQValueLossSeparateEntropy
from models import create_nets
 

import gym
import tensorflow as tf 

from parameters import observation_shape, action_shape, low_bound, upper_bound
from parameters import sac_args as args

from env_wrappers import obs2vec

import os 
import pickle



class SAC:
    
    def __init__(self):

        self.soft_tau = args.soft_tau
        self.device = 'cpu'


        policy_net, q_net_1, q_net_2, target_q_net_1, target_q_net_2, policy_optim,\
                        q1_optim, q2_optim = create_nets()
 
        self.policy_net = policy_net
        self.soft_q_net_1 = q_net_1
        self.soft_q_net_2 = q_net_2
        self.target_q_net_1 = target_q_net_1
        self.target_q_net_2 = target_q_net_2
        
        self.policy_optimizer = policy_optim
        self.q_optim_1 = q1_optim
        self.q_optim_2 = q2_optim

 
        self.q_loss = NStepQValueLossSeparateEntropy(\
                    args.gamma, args.q_weights, n_steps=args.n_step_loss, rescaling=args.rescaling)
 

        self.target_entropy = -action_shape # -22.

       
        self.sac_log_alpha = tf.Variable(initial_value=0, trainable=True, dtype=tf.float32)


        # self.sac_alpha = self.sac_log_alpha.exp().item()        
        self.sac_alpha = tf.math.exp(self.sac_log_alpha).numpy()

        
        # self.alpha_optim = torch.optim.Adam([self.sac_log_alpha], lr=1e-3)
        self.alpha_optim = tf.keras.optimizers.Adam(learning_rate=1e-3) 

        self.n_steps = args.n_step_train # 10.
        self.eta = args.priority_weight  # priority weight

    
        self.q_weights = tf.Variable(initial_value=[[args.q_weights + [1]]], 
                                    trainable=False, dtype=tf.float32) # (1, 1, q_dim)

        self.norm_obs = False

 
        self.i = 0
        self.prev_action = np.zeros((action_shape,)) 
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(22,), dtype=np.float32)


    def compute_mask(self, is_done): # is_done: (b, T).
        '''        
        if first done happend at t <= T-1, then mask[t+1:T] will all be 0.
        '''

        B = is_done.shape[0]

        # mask = tf.ones_like(is_done, dtype=tf.float32)
        # mask[:, 1:] = 1.0 - tf.cast(tf.math.cumsum(is_done[:, :-1], axis=-1) > 0, tf.float32)
        
        mask = 1.0 - tf.cast(tf.math.cumsum(is_done[:, :-1], axis=-1) > 0, tf.float32)
        ones = tf.ones((B, 1), dtype=tf.float32)

        mask = tf.concat([ones, mask], axis=1)

        return mask[:, -self.n_steps:]
    

 


    def sample_action_log_prob(self, observation_t):

        '''
        observation_t: (b, T+1, dim), T=10.
        '''

        # mean, log_std: (b, T+1, action_dim).
        mean, log_std = self.policy_net(observation_t)

        B, T1, a_dim = mean.shape
 
        std = tf.math.exp(log_std)
 
        distribution = tf.compat.v1.distributions.Normal(mean, std)

 
        z = distribution.sample() # (b, T+1, action_dim).
 
        action = tf.math.tanh(z) # (b, T+1, action_dim).


        log_prob = distribution.log_prob(z) # (b, T+1, action_dim).

        # calculate logarithms like a noob:
        # log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6)

        # # calculate logarithms like a pro: 
        log_prob = log_prob - math.log(4.0) + 2 * tf.math.log(\
                        tf.math.exp(z) + tf.math.exp(-z)) # (b, T+1, action_dim).

 
        log_prob = tf.reduce_sum(log_prob, axis=-1) # (b, T+1).

        return action, log_prob  # action: (b, T+1, action_dim), # log_prob: (b, T+1).
    



    def act(self, observation):                

        if(self.i % 4 == 0):            
            self.i = 1

            obs_vec = obs2vec(observation) 
            obs_vec = tf.convert_to_tensor(obs_vec[None, None, ...], dtype=tf.float32)
           
            action = self._act(obs_vec, False)[0, 0].numpy() # (22,)            
            # mean, _ = self.policy_net(obs_vec, training=False) # (b, T+1, action_dim).
            # action = tf.math.tanh(mean)[0, 0].numpy()            

            action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
            action = np.clip(action, low_bound, upper_bound)
            
            self.prev_action = action
  
        else:
            self.i += 1 
            
        return self.prev_action

        # return self.action_space.sample()




    def _act(self, observation_t, training=True): # no need grad.
        
        mean, log_std = self.policy_net(observation_t, 
                            training=training) # (b, T+1, action_dim).
        std = tf.math.exp(log_std)

        B = mean.shape[0] # b.


        dist = tf.compat.v1.distributions.Normal(mean, std)

        # action_t = tf.math.tanh(mean)
        # action_t[B // 2:] = tf.math.tanh(dist.sample()[B // 2:])
 
        action_t = tf.concat([tf.math.tanh(mean[:B // 2]),
                            tf.math.tanh(dist.sample()[B // 2:])], axis=0)

        return action_t



    def act_q(self, observation): # no need grad.
         
        observation_t = tf.convert_to_tensor(observation, dtype=tf.float32)

        # observation_t.unsqueeze_(1)
        observation_t = tf.expand_dims(observation_t, axis=1)

        action_t = self._act(observation_t)
        
        action_t = tf.squeeze(action_t, axis=1)

        return action_t.numpy()



    def batch_to_tensors(self, batch):
        def t(x): 
            return tf.convert_to_tensor(x, dtype=tf.float32)
 
        return map(t, batch)



    def calculate_priority(self, q_1_loss, q_2_loss, segment_length):
        '''
        q_1_loss, q_2_loss: (n_env, T). segment_length: (n_env,).
        ''' 
              
        q_loss = tf.math.sqrt(2.0 * tf.math.maximum(q_1_loss, q_2_loss)) # (n_env, T).

        # max_over_time = torch.max(q_loss, dim=1)[0] # (,)?! "[0]" should be removed?

        # print(f'\n\n\n###### q_loss.shape: {q_loss.shape}\n\n\n')
        max_over_time = tf.math.reduce_max(q_loss, axis=1)[0]
 
        mean_over_time = tf.reduce_sum(q_loss, axis=1) / segment_length # (n_env,).

        priority_loss = self.eta * max_over_time + (1 - self.eta) * mean_over_time # (n_env,).

        return (priority_loss + 1e-6).numpy() # (n_env,).



    def calculate_priority_loss(self, data): # 

        '''
            - no need grad, only used for segment sampler.
            - data:
                obs: (n_env, T+1, dim). act, rew: (n_env, T, dim). don: (n_env, T). T==10.
        '''

        # almost same as q_value_loss
        obs, actions, rewards, is_done = self.batch_to_tensors(data)

        # if first done for env i happend at t <= T-1, then mask[i, t+1:T] will all be 0.
        mask = self.compute_mask(is_done) # (n_env, T=10)
        
        # segment_length = mask.sum(-1) + 1 # (n_env,)
        segment_length = tf.reduce_sum(mask, axis=-1) + 1
      


        next_q, log_prob = self.pred_next_q(obs) # (b, T+1, q_dim+1), (b, T+1).
        
        next_ent = -self.sac_alpha * log_prob # (n_env, T+1).

        next_q = next_q[:, 1:]
        next_ent = next_ent[:, 1:]
        

        q_1_loss, q_2_loss = self.calc_q_value_loss( # (n_env, T), (n_env, T)
            obs, actions, rewards, is_done, mask, next_q, next_ent)

        
        priority_loss = self.calculate_priority(q_1_loss, q_2_loss, segment_length)  # (n_env,).
        return priority_loss # (n_env,).

    
    
    def pred_next_q(self, obs):
         
        # action: (n_env, T+1, action_dim).
        action, log_prob = self.sample_action_log_prob(obs)
         
        next_q_1 = self.target_q_net_1(obs, action) # (n_env, T+1, q_dim+1).
         
        next_q_2 = self.target_q_net_2(obs, action)


        next_q = tf.math.minimum(next_q_1, next_q_2) # (n_env, T+1, q_dim+1). 

         
        return next_q, log_prob # (n_env, T+1, q_dim+1), (n_env, T+1). 



    def calc_q_value_loss(self, obs, actions, rewards, is_done, mask, next_q, next_ent):
        '''
        - Need grad if called by learn_q_from_data(); No need if called by calculate_priority_loss().
        - T==10.
        - obs: (n_env, T+1, dim). actions, rewards: (n_env, T, q_dim). is_done: (n_env, T). 
        - mask: (n_env, T).
        ''' 
        

        next_q = next_q[:, -self.n_steps:] # no effect.
        next_ent = next_ent[:, -self.n_steps:]
        rewards = rewards[:, -self.n_steps:] # no effect.
        is_done = is_done[:, -self.n_steps:] # no effect.
        
        target_q_value = self.q_loss.compute_target_q(next_q, next_ent, rewards, is_done, mask)

        mask = tf.expand_dims(mask, axis=-1)

        current_q_1 = self.soft_q_net_1(obs[:, :-1], actions)[:, -self.n_steps:] # (n_env, T, q_dim+1).            
        q_1_loss = 0.5 * mask * (self.q_weights * ((current_q_1 - target_q_value) ** 2)) # (b, T, q_dim+1).   
        q_1_loss = tf.reduce_sum(q_1_loss, axis=-1) # (n_env, T). 
    
    
        current_q_2 = self.soft_q_net_2(obs[:, :-1], actions)[:, -self.n_steps:] # (n_env, T, q_dim+1).            
        q_2_loss = 0.5 * mask * (self.q_weights * ((current_q_2 - target_q_value) ** 2)) # (b, T, q_dim+1).             
        q_2_loss = tf.reduce_sum(q_2_loss, axis=-1) # (n_env, T).         

        return q_1_loss, q_2_loss

        # current_q_1 = self.soft_q_net_1(obs[:, :-1], actions) # (n_env, T, q_dim+1).
        # current_q_2 = self.soft_q_net_2(obs[:, :-1], actions) # (n_env, T, q_dim+1).
 
        # current_q_1 = current_q_1[:, -self.n_steps:] # no effect.
        # current_q_2 = current_q_2[:, -self.n_steps:] # no effect.
        # next_q = next_q[:, -self.n_steps:] # no effect.
        # next_ent = next_ent[:, -self.n_steps:]
        # rewards = rewards[:, -self.n_steps:] # no effect.
        # is_done = is_done[:, -self.n_steps:] # no effect.
 
        # q_1_loss = self.q_loss(current_q_1, next_q, next_ent, rewards, is_done, mask) # (n_env, T, dim). 
        # q_2_loss = self.q_loss(current_q_2, next_q, next_ent, rewards, is_done, mask) # (n_env, T, dim). 

        # q_1_loss = tf.reduce_sum(q_1_loss, axis=-1) # (n_env, T). 
        # q_2_loss = tf.reduce_sum(q_2_loss, axis=-1) # (n_env, T). 
        # return q_1_loss, q_2_loss



    def soft_target_update(self):
 
        new_weights = [] 
        for p, tp in zip(self.soft_q_net_1.get_weights(), self.target_q_net_1.get_weights()):
            new_weights.append((1.0 - self.soft_tau) * tp + self.soft_tau * p)        
        self.target_q_net_1.set_weights(new_weights)


        new_weights = [] 
        for p, tp in zip(self.soft_q_net_2.get_weights(), self.target_q_net_2.get_weights()):
            new_weights.append((1.0 - self.soft_tau) * tp + self.soft_tau * p)        
        self.target_q_net_2.set_weights(new_weights)


 
    def learn_q_from_data(self,
                          importance_weights,
                          obs, actions, rewards, is_done,
                          mask, 
                          segment_length, # (b,)
                          ):
        

        next_q, log_prob = self.pred_next_q(obs) # (b, T+1, q_dim+1), (b, T+1).

        next_ent = -self.sac_alpha * log_prob # (n_env, T+1).

        next_q = next_q[:, 1:]
        next_ent = next_ent[:, 1:]
 
 
        next_q = next_q[:, -self.n_steps:] # no effect.
        next_ent = next_ent[:, -self.n_steps:]
        rewards = rewards[:, -self.n_steps:] # no effect.
        is_done = is_done[:, -self.n_steps:] # no effect.
        
        target_q_value = self.q_loss.compute_target_q(next_q, next_ent, rewards, is_done, mask)


        mask = tf.expand_dims(mask, axis=-1)


        with tf.GradientTape() as tape: 
            current_q_1 = self.soft_q_net_1(obs[:, :-1], actions)[:, -self.n_steps:] # (n_env, T, q_dim+1).            
            q_1_loss = 0.5 * mask * (self.q_weights * ((current_q_1 - target_q_value) ** 2)) # (b, T, q_dim+1).   
            q_1_loss = tf.reduce_sum(q_1_loss, axis=-1) # (n_env, T). 

            q_1_loss_red = tf.math.reduce_mean((importance_weights * \
                        tf.reduce_sum(q_1_loss, axis=-1) / segment_length)) # (,)
        grads = tape.gradient(q_1_loss_red, self.soft_q_net_1.trainable_variables)
        self.q_optim_1.apply_gradients(zip(grads, self.soft_q_net_1.trainable_variables))
    

        with tf.GradientTape() as tape: 
            current_q_2 = self.soft_q_net_2(obs[:, :-1], actions)[:, -self.n_steps:] # (n_env, T, q_dim+1).            
            q_2_loss = 0.5 * mask * (self.q_weights * ((current_q_2 - target_q_value) ** 2)) # (b, T, q_dim+1).             
            q_2_loss = tf.reduce_sum(q_2_loss, axis=-1) # (n_env, T).         
            q_2_loss_red = tf.math.reduce_mean((importance_weights * \
                        tf.reduce_sum(q_2_loss, axis=-1) / segment_length)) # (,)            
        grads = tape.gradient(q_2_loss_red, self.soft_q_net_2.trainable_variables)
        self.q_optim_2.apply_gradients(zip(grads, self.soft_q_net_2.trainable_variables))
 


        priority = self.calculate_priority(q_1_loss, q_2_loss, segment_length) # (n_env,).

        return q_1_loss_red.numpy(), q_2_loss_red.numpy(), priority

 


    def learn_p_from_data(self, importance_weights, obs, mask, segment_length):
        '''
            obs: (b, T+1, dim).  
            segment_length: (b,), 
            mask: (b, T),
            T=10.      
        '''
                
        # policy_loss, alpha_loss: (b, T), q_min: (b, T+1, q_dim).                        

        with tf.GradientTape() as tape: 

            q_min, log_prob = self.pred_next_q(obs) # (b, T+1, q_dim+1), (b, T+1).
    
            target_log_prob = tf.reduce_sum(self.q_weights * q_min, axis=-1) # (b, T+1).

            log_prob = log_prob[:, -self.n_steps:] # (b, T).
            target_log_prob = target_log_prob[:, -self.n_steps:] # (b, T).

            policy_loss = mask * (self.sac_alpha * log_prob - target_log_prob) # (b, T),
        
            policy_loss = tf.math.reduce_mean((importance_weights * \
                        tf.reduce_sum(policy_loss, axis=-1) / segment_length)) # (,)
                     
        grads = tape.gradient(policy_loss, self.policy_net.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1)
        self.policy_optimizer.apply_gradients(zip(grads, self.policy_net.trainable_variables))
        
        mean_q_min = tf.math.reduce_mean((tf.reduce_sum(q_min, axis=1) / \
                            tf.expand_dims(segment_length, axis=-1)), axis=0) # (q_dim,)
 
        
        
        log_prob = tf.convert_to_tensor(log_prob.numpy(), dtype=tf.float32) # detach. # (b, T).
        with tf.GradientTape() as tape:    
            alpha_loss = -(self.sac_log_alpha * (log_prob + self.target_entropy)) * mask # (b, T).
            
            alpha_loss = tf.math.reduce_mean((importance_weights * \
                        tf.reduce_sum(alpha_loss, axis=-1) / segment_length)) # (,)
                     
        grads = tape.gradient(alpha_loss, self.sac_log_alpha) 
        self.alpha_optim.apply_gradients(zip([grads], [self.sac_log_alpha]))
       
        self.sac_alpha = tf.math.exp(self.sac_log_alpha).numpy()

        return policy_loss.numpy(), alpha_loss.numpy(), mean_q_min.numpy() # (,), (,), (q_dim,)




    def learn_from_data(self, data, importance_weights=1.0):
     
        # obs: (b, T+1, dim). actions, rewards: (b, T, q_dim). is_done: (b, T).  
        obs, actions, rewards, is_done = self.batch_to_tensors(data)
        mask = self.compute_mask(is_done) # (b, T=10)


        # adding 1 to segment_len everywhere is normal, because this will remove division by zero,
            # and the gradients at each time step will change equally
        segment_length = tf.reduce_sum(mask, axis=-1) + 1 # (b,)
        
        importance_weights = tf.convert_to_tensor(importance_weights, dtype=tf.float32)
  
        q_1_loss, q_2_loss, priority = self.learn_q_from_data(importance_weights,\
                     obs, actions, rewards, is_done, mask, segment_length) # (,), (,), (b,).
        
        self.soft_target_update()

        
        policy_loss, alpha_loss, q_min = self.learn_p_from_data(importance_weights, 
                                    obs, mask, segment_length) # (,), (,), (q_dim,).
   
        q_dim = rewards.shape[-1]
        mean_batch_reward = tf.reduce_sum(self.q_weights[:,:,:q_dim] * \
                                        rewards[:, -self.n_steps:, :], axis=-1)  # (b, T)

        mean_batch_reward = tf.math.reduce_mean(tf.reduce_sum(mask \
                            * mean_batch_reward, axis=-1) / segment_length).numpy() # (,)

        losses = np.array([policy_loss, alpha_loss, q_1_loss, q_2_loss, mean_batch_reward]) # (5,)
 
        return losses, q_min, priority # (5,), (q_dim,), (b,)




    def save(self, dir_name, name):
 
        if(not os.path.exists(dir_name)): 
            os.makedirs(dir_name, exist_ok=True)

        path = os.path.join(dir_name, name)

        state_dict = {}
 
        state_dict['policy_net'] = self.policy_net.get_weights()
        state_dict['soft_q_net_1'] = self.soft_q_net_1.get_weights()
        state_dict['soft_q_net_2'] = self.soft_q_net_2.get_weights()
        state_dict['target_q_net_1'] = self.target_q_net_1.get_weights()
        state_dict['target_q_net_2'] = self.target_q_net_2.get_weights()
        state_dict['sac_log_alpha'] = self.sac_log_alpha
        
 
        with open(path, 'wb') as f:
            pickle.dump(state_dict, f)             

        print(f'saved ckpt: {path}')



    def load(self, path):
        
        # path = os.path.join(dir_name, name)
        with open(path, 'rb') as f:            
            state_dict = pickle.load(f)                    
 
        self.policy_net.set_weights(state_dict['policy_net'])
        self.soft_q_net_1.set_weights(state_dict['soft_q_net_1'])
        self.soft_q_net_2.set_weights(state_dict['soft_q_net_2'])
        self.target_q_net_1.set_weights(state_dict['target_q_net_1'])
        self.target_q_net_2.set_weights(state_dict['target_q_net_2'])        
        # self.sac_log_alpha = state_dict['sac_log_alpha']
       
        print(f'loaded ckpt: {path}')
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



class SAC:
    
    def __init__(self):
 
        self.actor, self.critic1, self.critic2, self.critic1_tgt, self.critic2_tgt, \
                        self.actor_optim, self.critic1_optim, self.critic2_optim = create_nets()
  
        self.gammas = np.array([[args.gamma**i] for i in range(args.n_step_loss + 1)], dtype=np.float32)       
        self.q_weights = tf.Variable(initial_value=[[args.q_weights + [1]]], 
                                    trainable=False, dtype=tf.float32) # (1, 1, q_dim)
        
        self.sac_log_alpha = tf.Variable(initial_value=0, trainable=True, dtype=tf.float32)    
        self.sac_alpha = tf.math.exp(self.sac_log_alpha).numpy()
        self.alpha_optim = tf.keras.optimizers.Adam(learning_rate=1e-3) 

        self.i = 0
        self.prev_action = np.zeros((action_shape,)) 


    def action_rescale(self, action):
        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        return np.clip(action, low_bound, upper_bound)



    def compute_target_q(self,                 
                next_q_ent, # (b, T=10, q_dim+1). 
                ent, # (b, T). 
                reward,  # (b, T, q_dim). 
                is_done, # (b, T). 
                mask # (b, T).
            ):
 
        next_q_ent = next_q_ent.numpy() 
        ent = ent.numpy() 
        reward = reward.numpy() * mask
        is_done = is_done.numpy()[:,:,None]
        mask = mask.numpy()[:,:,None]

        B, T, q_dim = reward.shape # shape == (b, 10, q_dim)

      
        next_q_ent = mask * (1.0 - is_done) * inv_rescaling_fn(next_q_ent) # (b, T, q_dim+1). 

        next_q = next_q_ent[:,:,:-1] # (b, T, q_dim)
        next_ent = next_q_ent[:,:,-1] # (b, T)


        target_q_value = np.zeros([B, T, q_dim + 1], dtype=np.float32)

  
        for t1 in range(T): # T: 10.
            
            tn = min(T - 1, t1 + args.n_step_loss - 1) # args.n_step_loss: 5. 
            n = tn - t1 + 1

            # n-step return.               
            reward_sum = np.sum(reward[:, t1:t1+n, :] * self.gammas[:n], axis=1) # (b, q_dim) 
            target_q_value[:, t1, :-1] = reward_sum + self.gammas[n] * next_q[:, t1+n-1] # (b, q_dim). 

            ent_sum = np.sum(ent[:, t1:t1+n] * self.gammas[1:n+1, 0], axis=1)  # (b,)                    
            target_q_value[:, t1, -1] = ent_sum + self.gammas[n] * next_ent[:, t1+n-1] # (b,).             

 
        target_q_value = rescaling_fn(target_q_value) # (b, T, q_dim+1)

        return tf.convert_to_tensor(target_q_value, dtype=tf.float32)



    def compute_mask(self, is_done): # is_done: (b, T).
        '''        
        if first done happend at t <= T-1, then mask[t+1:T] will all be 0.
        '''

        B = is_done.shape[0]

        mask = 1.0 - tf.cast(tf.math.cumsum(is_done[:, :-1], axis=-1) > 0, tf.float32)
        ones = tf.ones((B, 1), dtype=tf.float32)

        mask = tf.concat([ones, mask], axis=1)

        return mask[:, -args.n_step_train:]
    

 


    def act(self, observation):                

        if(self.i % 4 == 0):            
            self.i = 1

            obs_vec = obs2vec(observation) 
            obs_vec = tf.convert_to_tensor(obs_vec[None, None, ...], dtype=tf.float32)
                    
            mean, _ = self.actor(obs_vec, training=False) # (b, T+1, action_dim).
        
            self.prev_action = self.action_rescale(tf.math.tanh(mean)[0, 0].numpy())
  
        else:
            self.i += 1 
            
        return self.prev_action
 


    def act_sampler(self, observation): # (n_env, 1, obs_dim)
         
        observation_t = tf.convert_to_tensor(observation, dtype=tf.float32) 

        mean, log_std = self.actor(observation_t, training=True) # (b, T+1, action_dim).
        std = tf.math.exp(log_std)
        dist = tf.compat.v1.distributions.Normal(mean, std)

        B = mean.shape[0] # b.
        action_t = tf.concat([tf.math.tanh(mean[:B // 2]),
                            tf.math.tanh(dist.sample()[B // 2:])], axis=0)
          
        return self.action_rescale(action_t.numpy())



    def batch_to_tensors(self, batch):
        def t(x): 
            return tf.convert_to_tensor(x, dtype=tf.float32)
 
        return map(t, batch)



    def calculate_priority(self, q_1_loss, q_2_loss, segment_length):
        '''
        q_1_loss, q_2_loss: (n_env, T). segment_length: (n_env,).
        ''' 
              
        q_loss = tf.math.sqrt(2.0 * tf.math.maximum(q_1_loss, q_2_loss)) # (n_env, T).

        max_over_time = tf.math.reduce_max(q_loss, axis=1)[0]
 
        mean_over_time = tf.reduce_sum(q_loss, axis=1) / segment_length # (n_env,).

    
        priority_loss = args.priority_weight * max_over_time + (1 - args.priority_weight) * mean_over_time # (n_env,).

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
      


        a, log_prob = self.actor_predict(obs) # (b, T+1, action_dim), # (b, T+1).
        next_q = self.critic_tgt_predict(obs, a) # (b, T+1, q_dim+1).


        
        next_ent = -self.sac_alpha * log_prob # (n_env, T+1).

        next_q = next_q[:, 1:]
        next_ent = next_ent[:, 1:]
        

        q_1_loss, q_2_loss = self.calc_q_value_loss( # (n_env, T), (n_env, T)
            obs, actions, rewards, is_done, mask, next_q, next_ent)

        
        priority_loss = self.calculate_priority(q_1_loss, q_2_loss, segment_length)  # (n_env,).
        return priority_loss # (n_env,).

    


    def calc_q_value_loss(self, obs, actions, rewards, is_done, mask, next_q, next_ent):
        '''        
        - T==10.
        - obs: (n_env, T+1, dim). actions, rewards: (n_env, T, q_dim). is_done: (n_env, T). 
        - mask: (n_env, T).
        ''' 
        

        next_q = next_q[:, -args.n_step_train:] # no effect.
        next_ent = next_ent[:, -args.n_step_train:]
        rewards = rewards[:, -args.n_step_train:] # no effect.
        is_done = is_done[:, -args.n_step_train:] # no effect.
        
        target_q = self.compute_target_q(next_q, next_ent, rewards, is_done, mask)

        mask = tf.expand_dims(mask, axis=-1)

        current_q_1 = self.critic1(obs[:, :-1], actions)[:, -args.n_step_train:] # (n_env, T, q_dim+1).            
        q_1_loss = 0.5 * mask * (self.q_weights * ((current_q_1 - target_q) ** 2)) # (b, T, q_dim+1).   
        q_1_loss = tf.reduce_sum(q_1_loss, axis=-1) # (n_env, T). 
    
    
        current_q_2 = self.critic2(obs[:, :-1], actions)[:, -args.n_step_train:] # (n_env, T, q_dim+1).            
        q_2_loss = 0.5 * mask * (self.q_weights * ((current_q_2 - target_q) ** 2)) # (b, T, q_dim+1).             
        q_2_loss = tf.reduce_sum(q_2_loss, axis=-1) # (n_env, T).         

        return q_1_loss, q_2_loss



    def update_critic_target(self):
 
        new_weights = [] 
        for p, tp in zip(self.critic1.get_weights(), self.critic1_tgt.get_weights()):
            new_weights.append((1.0 - args.soft_tau) * tp + args.soft_tau * p)        
        self.critic1_tgt.set_weights(new_weights)


        new_weights = [] 
        for p, tp in zip(self.critic2.get_weights(), self.critic2_tgt.get_weights()):
            new_weights.append((1.0 - args.soft_tau) * tp + args.soft_tau * p)        
        self.critic2_tgt.set_weights(new_weights)



    def critic_tgt_predict(self, obs, act):
        q_1 = self.critic1_tgt(obs, act) # (n_env, T+1, q_dim+1).         
        q_2 = self.critic2_tgt(obs, act)        
        return tf.math.minimum(q_1, q_2) # (n_env, T+1, q_dim+1).    


 
    def train_step_critic(self,
                          importance_weights,
                          obs, actions, rewards, is_done,
                          mask, segment_length, # (b,)
                          ):
        

        def optim_step(critic, optim, target_q):

            with tf.GradientTape() as tape: 
                current_q = critic(obs[:, :-1], actions)[:, -args.n_step_train:] # (n_env, T, q_dim+1).            
                q_loss = 0.5 * mask * (self.q_weights * ((current_q - target_q) ** 2)) # (b, T, q_dim+1).   
                q_loss = tf.reduce_sum(q_loss, axis=-1) # (n_env, T). 

                q_loss_mean = tf.math.reduce_mean((importance_weights * \
                            tf.reduce_sum(q_loss, axis=-1) / segment_length)) # (,)
                
            grads = tape.gradient(q_loss_mean, critic.trainable_variables)
            optim.apply_gradients(zip(grads, critic.trainable_variables))            
            return q_loss, q_loss_mean
        

        a, log_prob = self.actor_predict(obs[:, 1:]) # (b, T, action_dim), # (b, T).
        next_q = self.critic_tgt_predict(obs[:, 1:], a) # (b, T, q_dim+1).


        ent = -self.sac_alpha * log_prob # (n_env, T).
 
        next_q = next_q[:, -args.n_step_train:] # no effect.
        ent = ent[:, -args.n_step_train:]
        rewards = rewards[:, -args.n_step_train:] # no effect.
        is_done = is_done[:, -args.n_step_train:] # no effect.
        
        target_q = self.compute_target_q(next_q, ent, rewards, is_done, mask)


        mask = tf.expand_dims(mask, axis=-1)

        q1_loss, q1_loss_mean = optim_step(self.critic1, self.critic1_optim, target_q)
        q2_loss, q2_loss_mean = optim_step(self.critic2, self.critic2_optim, target_q)

        priority = self.calculate_priority(q1_loss, q2_loss, segment_length) # (n_env,).

        return q1_loss_mean.numpy(), q2_loss_mean.numpy(), priority

 





    def actor_predict(self, observation_t):

        '''
        observation_t: (b, T+1, dim), T=10.
        '''

        # mean, log_std: (b, T+1, action_dim).
        mean, log_std = self.actor(observation_t)

        std = tf.math.exp(log_std)
 
        distribution = tf.compat.v1.distributions.Normal(mean, std)

 
        z = distribution.sample() # (b, T+1, action_dim).
 
        action = tf.math.tanh(z) # (b, T+1, action_dim).

        # calculate logarithms like a noob:
        # log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6)
        
        log_prob = distribution.log_prob(z) # (b, T+1, action_dim).
 
        # # calculate logarithms like a pro: 
        log_prob = log_prob - math.log(4.0) + 2 * tf.math.log(\
                        tf.math.exp(z) + tf.math.exp(-z)) # (b, T+1, action_dim).

 
        log_prob = tf.reduce_sum(log_prob, axis=-1) # (b, T+1).

        return action, log_prob  # (b, T+1, action_dim), # (b, T+1).
    




    def train_step_actor(self, importance_weights, obs, mask, segment_length):
        '''
            obs: (b, T+1, dim).  
            segment_length: (b,), 
            mask: (b, T),
            T=10.      
        '''
                
        # policy_loss, alpha_loss: (b, T), q_min: (b, T+1, q_dim).                        

        with tf.GradientTape() as tape: 

            a, log_prob = self.actor_predict(obs) # (b, T+1, action_dim), # (b, T+1).
            q_value = self.critic_tgt_predict(obs, a) # (b, T+1, q_dim+1).
                
            q_value = tf.reduce_sum(self.q_weights * q_value, axis=-1) # (b, T+1).


            log_prob = log_prob[:, -args.n_step_train:] * mask # (b, T).
            q_value = q_value[:, -args.n_step_train:] * mask # (b, T).


            policy_loss = -(-self.sac_alpha * log_prob + q_value) # (b, T),
        
            policy_loss = tf.math.reduce_mean((importance_weights * \
                        tf.reduce_sum(policy_loss, axis=-1) / segment_length)) # (,)
                     
        grads = tape.gradient(policy_loss, self.actor.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1)
        self.actor_optim.apply_gradients(zip(grads, self.actor.trainable_variables))
        
  
        log_prob = tf.convert_to_tensor(log_prob.numpy(), dtype=tf.float32) # detach. # (b, T).
        ent_tgt = -action_shape
        with tf.GradientTape() as tape:    
            alpha_loss = -(self.sac_log_alpha * (log_prob + ent_tgt)) * mask # (b, T).
            
            alpha_loss = tf.math.reduce_mean((importance_weights * \
                        tf.reduce_sum(alpha_loss, axis=-1) / segment_length)) # (,)
                     
        grads = tape.gradient(alpha_loss, self.sac_log_alpha) 
        self.alpha_optim.apply_gradients(zip([grads], [self.sac_log_alpha]))
       
        self.sac_alpha = tf.math.exp(self.sac_log_alpha).numpy()

        return policy_loss.numpy(), alpha_loss.numpy() # (,), (,)




    def train_step(self, data, importance_weights):
     
        # obs: (b, T+1, dim). actions, rewards: (b, T, q_dim). is_done: (b, T).  
        obs, actions, rewards, is_done = self.batch_to_tensors(data)
        importance_weights = tf.convert_to_tensor(importance_weights, dtype=tf.float32)

        mask = self.compute_mask(is_done) # (b, T=10)

        # adding 1 to segment_len everywhere is normal, because this will remove division by zero,
            # and the gradients at each time step will change equally
        segment_length = tf.reduce_sum(mask, axis=-1) + 1 # (b,)        
  
        q_1_loss, q_2_loss, priority = self.train_step_critic(importance_weights,\
                     obs, actions, rewards, is_done, mask, segment_length) # (,), (,), (b,).
        
        self.update_critic_target()

        
        policy_loss, alpha_loss = self.train_step_actor(importance_weights, 
                                    obs, mask, segment_length) # (,), (,), (q_dim,).
   
        losses = np.array([policy_loss, alpha_loss, q_1_loss, q_2_loss]) # (4,)
 
        return losses, priority # (4,), (b,)




    def save(self, dir_name, name):
 
        if(not os.path.exists(dir_name)): 
            os.makedirs(dir_name, exist_ok=True)

        path = os.path.join(dir_name, name)

        state_dict = {}
 
        state_dict['policy_net'] = self.actor.get_weights()
        state_dict['soft_q_net_1'] = self.critic1.get_weights()
        state_dict['soft_q_net_2'] = self.critic2.get_weights()
        state_dict['target_q_net_1'] = self.critic1_tgt.get_weights()
        state_dict['target_q_net_2'] = self.critic2_tgt.get_weights()
        state_dict['sac_log_alpha'] = self.sac_log_alpha
        
 
        with open(path, 'wb') as f:
            pickle.dump(state_dict, f)             

        print(f'saved ckpt: {path}')



    def load(self, path):
        
        # path = os.path.join(dir_name, name)
        with open(path, 'rb') as f:            
            state_dict = pickle.load(f)                    
 
        self.actor.set_weights(state_dict['policy_net'])
        self.critic1.set_weights(state_dict['soft_q_net_1'])
        self.critic2.set_weights(state_dict['soft_q_net_2'])
        self.critic1_tgt.set_weights(state_dict['target_q_net_1'])
        self.critic2_tgt.set_weights(state_dict['target_q_net_2'])        
        self.sac_log_alpha = state_dict['sac_log_alpha']
       
        print(f'loaded ckpt: {path}')
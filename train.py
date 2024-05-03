
from collections import deque, namedtuple
import time

import numpy as np
import tensorflow as tf

from memory import Memory
from model import MLP, FlowPolicy
from explore import DisagreementExploration
  




class V():

    def __init__(self, value_hidden_sizes, lr, tau):  
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)     
        self.tau = tau   
        
        self.v = MLP('v', hidden_sizes=value_hidden_sizes, output_size=1)
        self.v_tgt = MLP('v_tgt', hidden_sizes=value_hidden_sizes, output_size=1)

  
    def train_step(self, obs, target_v):          
        with tf.GradientTape() as tape: 
            v = self.v(obs)  
            loss = tf.math.reduce_mean(tf.norm(v - target_v, axis=1, ord=1), axis=0)
     
        grads = tape.gradient(loss, self.v.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.v.trainable_variables))
  
    
    def update_target(self): 
        for tgt, src in zip(self.v_tgt.trainable_variables, self.v.trainable_variables):
            tf.assign(tgt, (1 - self.tau) * tgt + self.tau * src)
 
    def predict(self, obs):
        return self.v_tgt(obs)





class Q():
     
    def __init__(self, name, q_hidden_sizes, lr):   
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.q = MLP(name, hidden_sizes=q_hidden_sizes, output_size=output_size)
         
    def train_step(self, obs, act, target_q): 
        with tf.GradientTape() as tape:  
            q = self.q(tf.concat([obs, act], 1))         
            loss = tf.reduce_mean(tf.norm(q - target_q, axis=1, ord=1), axis=0) 
                 
        grads = tape.gradient(loss, self.q.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q.trainable_variables))
   
    def predict(self, obs, act):
        return self.q(tf.concat([obs, act], 1))



class DoubleQ():
     
    def __init__(self, q_hidden_sizes, lr):   
        self.q1 = Q('q1', q_hidden_sizes, lr)
        self.q2 = Q('q2', q_hidden_sizes, lr)
         
    def train_step(self, obs, act, target_q): 
        self.q1.train_step(obs, act, target_q)  
        self.q2.train_step(obs, act, target_q) 
        
    def predict(self, obs, act)
        q = tf.math.minimum(self.q1.predict(obs, act), self.q2.predict(obs, act))
        return q




class SAC:

    # args.
    def __init__(self, observation_shape, action_shape, v_tgt_field_size, args):  

        super().__init__(name='')

        self.args = args
 
        self.v = V(args.value_hidden_sizes, args.lr, args.tau)

        self.doubleQ = DoubleQ(args.q_hidden_sizes, args.lr)
 
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=args.policy_lr)   
        
        self.policy = FlowPolicy('policy', hidden_sizes=args.policy_hidden_sizes, output_size=action_shape[0], output_kernel_initializer='zeros', output_bias_initializer='zeros')

        self.beta = tf.Variable(tf.zeros(1), trainable=True, dtype=tf.float32, name='beta')     
        self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)      
        self.target_entropy = - 1 * np.sum(action_shape)


    # def train_step(self, obs, act, rew, don, obs_next):  
    def train_step(self, batch):   

        obs, act, rew, don, obs_next = batch
 

        # train policy.
        with tf.GradientTape() as tape: 
            act_pred, log_pis = self.policy(obs)  # (n_samples, batch_size, action_dim) and (...,1)
            act_pred, log_pis = tf.squeeze(act_pred, 0), tf.squeeze(log_pis, 0) 
            q_at_policy_action = self.doubleQ.predict(obs, act_pred) 
            policy_loss = tf.reduce_mean(alpha * log_pis - q_at_policy_action)
          
        grads = tape.gradient(policy_loss, self.policy.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))

        # train v.
        self.v.train_step(obs, -policy_loss)

        # train doubleQ.
        next_v = self.v.predict(obs_next)
        target_q = rew + self.args.discount * next_v * (1 - don)
        self.doubleQ.train_step(obs, act, target_q)
 

        with tf.GradientTape() as tape: 
            alpha = tf.math.exp(self.beta)
            alpha_loss = tf.reduce_mean(- alpha * log_pis - alpha * self.target_entropy)
   
        grads = tape.gradient(alpha_loss, self.beta.trainable_variables)
        self.alpha_optimizer.apply_gradients(zip(grads, self.beta.trainable_variables))



    def get_actions(self, obs):
        act, _ = self.policy(obs, self.args.n_sample_actions)
        return act
  

    def get_action_value(self, obs, act):
        return self.doubleQ.predict(obs, act)
        
    def update_target_net(self):
        self.v.update_target()






 


def learn(env, args, expl_args, alg_args):

    
    n_train_steps_per_env_step = alg_args.pop('n_train_steps_per_env_step', 1)
    
    
    episode_lengths = np.zeros((env.num_envs, 1), dtype=int)

   
    # env.observation_space.shape: (102,),  env.action_space.shape: (22,).
    memory = Memory(args.memory_size, env.observation_space.shape, env.action_space.shape)
    agent = SAC(env.observation_space.shape, env.action_space.shape, env.v_tgt_field_size alg_args)
    exploration = DisagreementExploration(env.observation_space.shape, env.action_space.shape, expl_args)
 
  
    memory.initialize(env, args.n_prefill_steps // env.num_envs)
 
    obs = env.reset() # obs.shape: (n_env, 102).


    for t in range(int(1e8)):
       
        actions = agent.get_actions(obs)  # (n_samples, n_env, action_dim=22)
        actions = exploration.select_best_action(obs, actions) # (n_env, action_dim=22)
        next_obs, r, done, info = env.step(actions) # r: (n_env, 1), done: (n_env, 1)
        r_aug = np.vstack([i.get('rewards', 0) for i in info]) # (n_env, 1)
        r_bonus = exploration.get_exploration_bonus(obs, actions, next_obs) # encourage exploration.
        
        # don't count "reaching max_episode_length (time limit)" as done.
        done_bool = np.where(episode_lengths + 1 == args.max_episode_length, np.zeros_like(done), done)  

        memory.store_transition(obs, actions, r + r_bonus + r_aug, done_bool, next_obs)
        obs = next_obs
   
        episode_lengths += 1

        # end of episode -- when all envs are done or max_episode length is reached, reset
        if any(done):
            # set episode_lengths of those whose done is True to 0.
            for d in np.nonzero(done)[0]:                                     
                episode_lengths[d] = 0
 


        batch = memory.sample(args.batch_size)
        agent.train_step(batch)
        agent.update_target_net()

        expl_loss = exploration.train(memory, args.batch_size)
 
    return agent
 
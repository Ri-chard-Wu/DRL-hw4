
from collections import deque, namedtuple
import time

import numpy as np
import tensorflow as tf

from memory import Memory
from model import MLP, FlowPolicy
from explore import DisagreementExploration
  




class Q():
     
    def __init__(self, name, q_hidden_sizes, output_size, lr):  
 
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


class V():

    def __init__(self, value_hidden_sizes, output_size, lr, tau):  
 

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




class DoubleQ():
     
    def __init__(self, q_hidden_sizes, output_size, lr):  
 
        self.q1 = Q('q1', q_hidden_sizes, output_size, lr)
        self.q2 = Q('q2', q_hidden_sizes, output_size, lr)
        

  
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

        self.seq = []

        self.value_function = MLP('value_function', hidden_sizes=args.value_hidden_sizes, output_size=1)
        self.target_value_function = MLP('target_value_function', hidden_sizes=argsvalue_hidden_sizes, output_size=1)

        self.doubleQ = DoubleQ()

        self.policy = FlowPolicy('policy', hidden_sizes=args.policy_hidden_sizes, output_size=action_shape[0], output_kernel_initializer='zeros', output_bias_initializer='zeros')


    def target_q(self, rew, obs_next, don):
        next_v = self.target_value_function(obs_next)
        q = rew + self.args.discount * next_v * (1 - don)
        return q


    def train_step(self, target_q):
        next_v = self.target_value_function(obs_next)
        q = rew + self.args.discount * next_v * (1 - don)
        return q





class SAC:
    def __init__(self, observation_shape, action_shape, v_tgt_field_size, *,
                    policy_hidden_sizes, q_hidden_sizes, value_hidden_sizes, alpha, discount, tau, lr, policy_lr, n_sample_actions=1, learn_alpha=True, use_nf=True, loss_ord=1):
        
        # inputs
        self.obs_ph = tf.placeholder(tf.float32, [None, *observation_shape], name='obs')
        self.actions_ph = tf.placeholder(tf.float32, [None, *action_shape], name='actions')
        self.rewards_ph = tf.placeholder(tf.float32, [None, 1], name='rewards')
        self.dones_ph = tf.placeholder(tf.float32, [None, 1], name='dones')
        self.next_obs_ph = tf.placeholder(tf.float32, [None, *observation_shape], name='next_obs')
 
        value_function = MLP('value_function', hidden_sizes=value_hidden_sizes, output_size=1)
        target_value_function = MLP('target_value_function', hidden_sizes=value_hidden_sizes, output_size=1)

        # Dual Q-func to avoid over-estimation problem?
        q1 = MLP('q1', hidden_sizes=q_hidden_sizes, output_size=1)
        q2 = MLP('q2', hidden_sizes=q_hidden_sizes, output_size=1)
         
        policy = FlowPolicy('policy', hidden_sizes=policy_hidden_sizes, output_size=action_shape[0], output_kernel_initializer='zeros', output_bias_initializer='zeros')
        
 
        if learn_alpha:
            beta = tf.Variable(tf.zeros(1), trainable=True, dtype=tf.float32, name='beta')
            alpha = tf.exp(beta)


        # 2. loss
        #   q value target
        next_v = target_value_function(self.next_obs_ph)
        target_q = self.rewards_ph + discount * next_v * (1 - self.dones_ph)




        #   q values loss terms
        q1_at_memory_action = q1(tf.concat([self.obs_ph, self.actions_ph], 1))
        q2_at_memory_action = q2(tf.concat([self.obs_ph, self.actions_ph], 1))
        self.q_at_memory_action = tf.minimum(q1_at_memory_action, q2_at_memory_action)

        
        q1_loss = tf.reduce_mean(tf.norm(q1_at_memory_action - target_q, axis=1, ord=loss_ord), axis=0)
        q2_loss = tf.reduce_mean(tf.norm(q2_at_memory_action - target_q, axis=1, ord=loss_ord), axis=0)



        #   policy loss term
        actions, log_pis = policy(self.obs_ph)  # (n_samples, batch_size, action_dim) and (...,1)
        actions, log_pis = tf.squeeze(actions, 0), tf.squeeze(log_pis, 0)
        q_at_policy_action = tf.minimum(q1(tf.concat([self.obs_ph, actions], 1)),
                                        q2(tf.concat([self.obs_ph, actions], 1)))
        policy_loss = tf.reduce_mean(alpha * log_pis - q_at_policy_action)



        #   value function loss term
        v = value_function(self.obs_ph) # MLP.
        target_v = q_at_policy_action - alpha * log_pis
        v_loss = tf.reduce_mean(tf.norm(v - target_v, axis=1, ord=loss_ord), axis=0)



        #   alpha loss term
        if learn_alpha:
            target_entropy = - 1 * np.sum(action_shape)
            alpha_loss = tf.reduce_mean(- alpha * log_pis - alpha * target_entropy)


        # 3. update ops
        
        policy_optimizer = tf.train.AdamOptimizer(policy_lr, name='policy_optimizer')
        policy_train_op = policy_optimizer.minimize(policy_loss, var_list=policy.trainable_vars, 
                                                    global_step=tf.train.get_or_create_global_step())
        self.train_ops = [policy_train_op]

 
        with tf.control_dependencies([policy_train_op]):
            flow_train_op = policy_optimizer.minimize(policy_loss, var_list=policy.flow_trainable_vars)
            self.train_ops += [flow_train_op]


        optimizer = tf.train.AdamOptimizer(lr, name='optimizer')
        with tf.control_dependencies([policy_train_op]):
            v_train_op = optimizer.minimize(v_loss, var_list=value_function.trainable_vars)
            q1_train_op = optimizer.minimize(q1_loss, var_list=q1.trainable_vars)
            q2_train_op = optimizer.minimize(q2_loss, var_list=q2.trainable_vars)


        #   combined train ops
        self.train_ops += [v_train_op, q1_train_op, q2_train_op]
        if learn_alpha:
            with tf.control_dependencies(self.train_ops):
                alpha_train_op = optimizer.minimize(alpha_loss, var_list=[beta])
            self.train_ops += [alpha_train_op]

        #   target value fn update
        self.target_update_ops = tf.group([tf.assign(target, (1 - tau) * target + tau * source) for target, source in zip(target_value_function.trainable_vars, value_function.trainable_vars)])

        # 4. get action op
        self.actions, _ = policy(self.obs_ph, n_sample_actions)


    def initialize(self, sess):
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())


    def get_actions(self, obs):
        return self.sess.run(self.actions, {self.obs_ph: np.atleast_2d(obs)})

    def get_action_value(self, obs, actions):
        return self.sess.run(self.q_at_memory_action, {self.obs_ph: obs, self.actions_ph: actions})

    def update_target_net(self):
        self.sess.run(self.target_update_ops)

    def train_step(self, batch):
        self.sess.run(self.train_ops, {self.obs_ph: batch.obs, 
                                       self.actions_ph: batch.actions, 
                                       self.rewards_ph: batch.rewards,
                                       self.dones_ph: batch.dones, 
                                       self.next_obs_ph: batch.next_obs})








def learn(env, args, expl_args, alg_args):

    
    n_train_steps_per_env_step = alg_args.pop('n_train_steps_per_env_step', 1)
    
    
    episode_lengths = np.zeros((env.num_envs, 1), dtype=int)

   
    memory = Memory(args.memory_size, env.observation_space.shape, env.action_space.shape)
    agent = SAC(env.observation_space.shape, env.action_space.shape, env.v_tgt_field_size, **alg_args)
    exploration = DisagreementExploration(env.observation_space.shape, env.action_space.shape, **expl_args)

    # initialize session, agent, saver
    sess = tf.get_default_session()
    agent.initialize(sess)
    
    exploration.initialize(sess, env=env)


    sess.graph.finalize()
    


   
    # init memory and env for training
    memory.initialize(env, args.n_prefill_steps // env.num_envs, training=(n_total_steps > 0), 
                      policy=agent if args.load_path else None)
    obs = env.reset()

    for t in range(int(1e8)):
      
      
        actions = agent.get_actions(obs)  # (n_samples, batch_size, action_dim)
        actions = exploration.select_best_action(obs, actions)
        next_obs, r, done, info = env.step(actions)
        r_aug = np.vstack([i.get('rewards', 0) for i in info]) # auxiliary?
        r_bonus = exploration.get_exploration_bonus(obs, actions, next_obs) # encourage exploration.
        
        # don't count "reaching max_episode_length (time limit)" as done.
        done_bool = np.where(episode_lengths + 1 == args.max_episode_length, np.zeros_like(done), done)  

        memory.store_transition(obs, actions, r + r_bonus + r_aug, done_bool, next_obs)
        obs = next_obs
   
        episode_lengths += 1

        # end of episode -- when all envs are done or max_episode length is reached, reset
        if any(done):
            for d in np.nonzero(done)[0]:                                     
                episode_lengths[d] = 0
 


        batch = memory.sample(args.batch_size)
        agent.train_step(batch)
        agent.update_target_net()

        expl_loss = exploration.train(memory, args.batch_size)
 
    return agent
 
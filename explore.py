import numpy as np
import tensorflow as tf

from model import MLP
from env_wrappers import RewardAugEnv



 
class DisagreementExploration():
    '''
    https://arxiv.org/abs/1906.04161, cite: 385.
    
    '''
    def __init__(self, observation_shape, action_shape, args):
        '''
        observation_shape: (102,),  action_shape: (22,).
        '''        

        self.observation_shape = observation_shape
        self.action_shape = action_shape


        # subset of obs indices to model / use in exploration modules
        # v_tgt_field_size: 5.
        # observation_shape[0]: the new obs shape after shriking v field size from
            # 2*11*11 to 5 in one env wrapper.
        self.v_tgt_field_size = observation_shape[0] - (339 - 2*11*11)  # L2M original obs dims are 339 where 2*11*11 is the orig vtgt field size
        
        # pose_idxs start with pelvis at 0 index (obs vector after the v_tgt_field)
        self.pose_idxs = np.array([*list(range(9)),                        # pelvis       (9 obs)
                                   *list(range(12,16)),   # joints r leg (4 obs)
                                   *list(range(20+3*11+3,20+3*11+3+4))]) # joints l leg (4 obs)
        self.pose_idxs += self.v_tgt_field_size  # offset for v_tgt_field
        
        self.idxs = np.hstack([list(range(self.v_tgt_field_size)), self.pose_idxs]) 

        # self.idxs: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 17, 18, 19, 20, 61, 62, 63, 64])  
        # self.idxs: (22,)



        self.lr = args.lr
        self.bonus_scale = args.bonus_scale


 
        # build graph
        # 1. networks
        self.state_predictors = [MLP(f'state_predictor_{i}', 
                                  hidden_sizes=args.state_predictor_hidden_sizes,
                                    output_size=len(self.idxs)) for i in range(args.n_state_predictors)]


        self.optimizer = [tf.keras.optimizers.Adam(learning_rate=args.lr) for _ in range(args.n_state_predictors)]


    def pre_process(self, obs):
        # self.idxs: (22,)   
        obs = tf.gather(obs, self.idxs, axis=1) # (n_env, 22) 
        obs_mean = tf.math.reduce_mean(obs, axis=0) # (22,).
        obs_std = tf.math.reduce_std(obs, axis=0) # (22,).
       
        normed_obs = (obs - obs_mean) / (obs_std + 1e-8) # (22,).
         
        return normed_obs, obs_mean, obs_std


    def predict(self, obs, act): # obs = (n_env, 102)

        # self.idxs: (22,)   
        # obs = tf.gather(obs, self.idxs, axis=1) # (n_env, 22) 
        # obs_mean = tf.math.reduce_mean(obs, axis=0) # (22,).
        # obs_std = tf.math.reduce_std(obs, axis=0) # (22,).
       
        # normed_obs = (obs - obs_mean) / (obs_std + 1e-8) # (22,).
             
        normed_obs, obs_mean, obs_std = self.pre_process(obs)

        normed_pred_next_obs = [model(tf.concat([normed_obs, act], 1)) for model in self.state_predictors] # [(n_env, 22), (n_env, 22), ...]
 
        return normed_pred_next_obs, obs_mean, obs_std


    def get_exploration_bonus(self, obs, act): 
        
        # obs: (n_env, obs_dim) == (n_env, 102).
        # act: (n_env, act_dim).
 
         
        normed_pred_next_obs, obs_mean, obs_std = self.predict(obs, act)
        normed_pred_next_obs = tf.stack(normed_pred_next_obs, axis=1).numpy()  # (B, n_state_predictors, 22)

        return self.bonus_scale * np.var(normed_pred_next_obs, axis=(1,2))[:,None] # (B, 1)?

 
    def select_best_action(self, obs, act):

        # obs = (n_env, 104); actions = (n_samples, n_env, action_dim)
        n_samples, n_env, action_dim = act.shape

        # reshape inputs to (n_samples*n_env, *_dim)
        obs = np.tile(obs, (n_samples, 1, 1)).reshape(-1, obs.shape[-1])
        act = act.reshape(-1, action_dim)


        obs = tf.convert_to_tensor(obs, tf.float32)
        act = tf.convert_to_tensor(act, tf.float32)

 
        normed_pred_next_obs, obs_mean, obs_std = self.predict(obs, act)
        normed_pred_next_obs = tf.stack(normed_pred_next_obs, axis=1).numpy()  # (B, n_state_predictors, 22)

        pred_next_obs = normed_pred_next_obs * (obs_std[None,None,:] + 1e-8)\
                                                + obs_mean[None,None,:]  # (B, n_state_predictors, obs_dim)
 
        pred_next_obs = pred_next_obs.numpy()

        # compute reward, split into 9 subarrays over the last axis.
        height, pitch, roll, dx, dy, dz, dpitch, droll, dyaw = np.split(
                        pred_next_obs[:,:,self.v_tgt_field_size: self.v_tgt_field_size+9], 9, -1)

        # split to produce 3 arrays of shape (n,), (n,) and (1,)  where n is half the pooled v_tgt_field
        # split to produce 3 arrays: [0:1], [1:self.v_tgt_field_size - 1], [self.v_tgt_field_size - 1:].
        x_vtgt_onehot, _, goal_dist = np.split(pred_next_obs[:,:,:self.v_tgt_field_size], [1, self.v_tgt_field_size - 1], axis=-1)  
        
        rewards = RewardAugEnv.compute_rewards(x_vtgt_onehot, goal_dist, height, pitch, roll, dx, dy, dz, dpitch, droll, dyaw, None, None, None, None, None, None)
        
        rewards = np.sum([v for v in rewards.values()], 0)  # (n_samples*n_env, n_state_predictors, 1)
        rewards = np.reshape(rewards, [n_samples, n_env, -1, 1])  # (n_samples, n_env, n_state_predictors, 1)
        rewards = np.sum(rewards, 2)  # sum over state predictors; out (n_samples, n_env, 1)

        actions = actions.reshape([n_samples, n_env, action_dim])

        # rewards.argmax(0)[None,...]: (1, n_env, 1)
        best_actions = np.take_along_axis(actions, rewards.argmax(0)[None,...], 0)  # out (1, n_env, action_dim)

        return np.squeeze(best_actions, 0)



    def train(self, memory, batch_size):
  
        for i in range(len(self.state_predictors)):

            model = self.state_predictors[i]

            obs, act, _, _, obs_next = memory.sample(batch_size)
             
            normed_obs, _, _ = self.pre_process(obs)  
            normed_next_obs, _, _ = self.pre_process(obs_next)                    

            with tf.GradientTape() as tape: 
                normed_pred_next_obs = model(tf.concat([normed_obs, act], 1))

                loss = tf.keras.losses.mse(normed_pred_next_obs, normed_next_obs)
    
            grads = tape.gradient(loss, model.trainable_variables)
            self.optimizer[i].apply_gradients(zip(grads, model.trainable_variables))

                
                 
 
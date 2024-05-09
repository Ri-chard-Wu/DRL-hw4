
 


import time
import gym
import numpy as np
 
from osim.env import L2M2019Env 

from parameters import train_env_args
 


 
import numpy as np

import os 
import pickle 
import tensorflow as tf 


class AttrDict(dict):
    def __getattr__(self, key):
        if key in self:
            value = self[key]
            if isinstance(value, dict):
                return DotDict(value)
            return value
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")




actor_args = AttrDict({
    "hidden_dim": 1024,
    "noisy": "False",
    "layer_norm": True,
    "afn": "elu",
    "residual": True,
    "dropout": 0.1,
    "lr": 3e-5,
    "normal": "True"
})
 


observation_shape = [2 * 11 * 11, 97]
action_shape = 22
low_bound = np.zeros((action_shape,))
upper_bound = np.ones((action_shape,))   

 
  

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

afns = {
    'relu': tf.keras.layers.ReLU,
    'elu': tf.keras.layers.ELU
}
 
     


def obs2vec(obs):
  
    def leg_to_numpy(leg):
        observation = []
        for k, v in leg.items():
            if type(v) is dict:
                observation += list(v.values())
            else:
                observation += v
            
        return np.array(observation)
 
 
    v_tgt_field = obs['v_tgt_field'].reshape(-1) / 10 # (242,)

    p = obs['pelvis']
    pelvis = np.array([p['height'], p['pitch'], p['roll']] + p['vel']) # (9,)

    r_leg = leg_to_numpy(obs['r_leg'])  # (44,)
    l_leg = leg_to_numpy(obs['l_leg'])  # (44,)

    flatten_observation = np.concatenate([v_tgt_field, pelvis, r_leg, l_leg]) # (339,)
    return flatten_observation # (339,)



class Layer(tf.keras.Model):

    def __init__(self, in_features, out_features, layer_norm, afn, residual=True, drop=0.0):        
        super().__init__()
 
        seq = []

        seq.append(tf.keras.layers.Dense(out_features))
        
        if layer_norm:
            seq.append(tf.keras.layers.LayerNormalization(epsilon=1e-5))

        if afn is not None:
            seq.append(afns[afn]())
 
        if drop != 0.0:
            seq.append(tf.keras.layers.Dropout(drop))

        self.seq = tf.keras.Sequential(seq)

        self.residual = residual and in_features == out_features


    def call(self, x_in, training=True):
         
        x = self.seq(x_in, training=training)
 
        if self.residual:
            x = x + x_in

        return x
 
class PolicyNet(tf.keras.Model):

    def __init__(self, args=actor_args):
        super().__init__() 

        h = args.hidden_dim
        ln = args.layer_norm
        afn = args.afn
        res = args.residual
        drop = args.dropout
         
        tgt_dim, obs_dim = observation_shape

        self.seq = tf.keras.Sequential([
            Layer(obs_dim + tgt_dim, h, ln, afn, res, drop),
            Layer(h, h, ln, afn, res, drop),
            Layer(h, h, ln, afn, res, drop),
        ])

        self.mean = Layer(h, action_shape, False, None)
        self.log_sigma = Layer(h, action_shape, False, None)

        self(tf.random.uniform((1, 1, obs_dim + tgt_dim)))


    def call(self, x, training=True):
        
        x = self.seq(x, training=training)

        mean = self.mean(x, training=training)

        log_sigma = self.log_sigma(x, training=training) 
        log_sigma = tf.clip_by_value(log_sigma, LOG_SIG_MIN, LOG_SIG_MAX)

        return mean, log_sigma



class Agent:
    
    def __init__(self):
 
        self.policy_net = PolicyNet()

        self.load('111022533_hw4_data')
 
        self.i = 0
        self.prev_action = np.zeros((action_shape,)) 
        
 



    def act(self, observation):                
        # obs_vec = obs2vec(observation) 
        obs_vec = tf.convert_to_tensor(observation[None, None, ...], dtype=tf.float32)
        
        # action = self._act(obs_vec, False)[0, 0].numpy() # (22,)            

        mean, _ = self.policy_net(obs_vec, training=False) # (b, T+1, action_dim).
        action = tf.math.tanh(mean)[0, 0].numpy()
        

        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)
        return action
        # self.prev_action = action

        
        # if(self.i % 4 == 0):            
        #     self.i = 1

        #     # obs_vec = obs2vec(observation) 
        #     obs_vec = tf.convert_to_tensor(observation[None, None, ...], dtype=tf.float32)
           
        #     # action = self._act(obs_vec, False)[0, 0].numpy() # (22,)            

        #     mean, _ = self.policy_net(obs_vec, training=False) # (b, T+1, action_dim).
        #     action = tf.math.tanh(mean)[0, 0].numpy()
            

        #     action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        #     action = np.clip(action, low_bound, upper_bound)
            
        #     self.prev_action = action
  
        # else:
        #     self.i += 1 
            
        # return self.prev_action  

 

  
    def load(self, path):
         
        with open(path, 'rb') as f:            
            state_dict = pickle.load(f)                    
 
        self.policy_net.set_weights(state_dict['policy_net'])
 
 




 

def obs2vec(obs):
  
    def leg_to_numpy(leg):
        observation = []
        for k, v in leg.items():
            if type(v) is dict:
                observation += list(v.values())
            else:
                observation += v
            
        return np.array(observation)
 
 
    v_tgt_field = obs['v_tgt_field'].reshape(-1) / 10 # (242,)

    p = obs['pelvis']
    pelvis = np.array([p['height'], p['pitch'], p['roll']] + p['vel']) # (9,)

    r_leg = leg_to_numpy(obs['r_leg'])  # (44,)
    l_leg = leg_to_numpy(obs['l_leg'])  # (44,)

    flatten_observation = np.concatenate([v_tgt_field, pelvis, r_leg, l_leg]) # (339,)
    return flatten_observation # (339,)



class SkeletonWrapper(gym.Wrapper):


    def __init__(self, env, is_train_env, args):

        gym.Wrapper.__init__(self, env)
        self.is_train_env = is_train_env

        self.frame_skip = args.frame_skip

        env.spec.timestep_limit = args.timestep_limit
        self.timestep_limit = args.timestep_limit
        self.episode_len = 0

        # reward weights from the environment
    

        self.alive_bonus = args.alive_bonus
        self.death_penalty = args.death_penalty
        self.task_bonus = args.task_bonus

        # placeholders
        self.step_time = 0
        self.v_tgt_step_penalty = 0

 
  



    def reset(self, **kwargs):
        obs = self.env.reset()

        # obs = self.observation_to_numpy(obs) # (339,)
        obs = obs2vec(obs) # (339,)
 
    
        self.episode_len = 0

        return obs # (339,)



    def step(self, action):

        raw_obs, reward, done, info = self.env.step(action)

        # obs = self.observation_to_numpy(raw_obs) # (339,)
        obs = obs2vec(raw_obs) # (339,)
 
        reward = self.shape_reward(action, reward, done)

        self.episode_len += 1
  
        return obs, reward, done, info



    @staticmethod
    def crossing_legs_penalty(state_desc):
        # stolen from Scitator
        pelvis_xyz = np.array(state_desc['body_pos']['pelvis'])
        left = np.array(state_desc['body_pos']['toes_l']) - pelvis_xyz
        right = np.array(state_desc['body_pos']['toes_r']) - pelvis_xyz
        axis = np.array(state_desc['body_pos']['head']) - pelvis_xyz
        cross_legs_penalty = np.cross(left, right).dot(axis)

        if cross_legs_penalty > 0:
            cross_legs_penalty = 0.0

        return 10 * cross_legs_penalty


   
    @staticmethod
    def get_v_body(state_desc):
        dx = state_desc['body_vel']['pelvis'][0]
        dy = state_desc['body_vel']['pelvis'][2]

        # print(f"\n\nstate_desc['body_vel']['pelvis']: {state_desc['body_vel']['pelvis']}\n\n")

        return np.asarray([dx, -dy])



    def get_v_tgt(self, state_desc):
        p_body = [state_desc['body_pos']['pelvis'][0], -state_desc['body_pos']['pelvis'][2]]
        v_tgt = self.vtgt.get_vtgt(p_body).T
        # print(f'\n\nv_tgt: {v_tgt}\n\n')

        # print(f"\n\nstate_desc['body_pos']['pelvis']: {state_desc['body_pos']['pelvis']}\n\n")

        return v_tgt



    @staticmethod
    def pelvis_velocity_bonus(v_tgt, v_body):
        v_tgt_abs = (v_tgt ** 2).sum() ** 0.5
        v_body_abs = (v_body ** 2).sum() ** 0.5
        v_dp = np.dot(v_tgt, v_body)
        cos = v_dp / (v_tgt_abs * v_body_abs + 0.1)
        bonus = v_body_abs 
        return (np.sign(cos) * bonus)[0]



    @staticmethod
    def target_achieve_bonus(v_tgt):
        v_tgt_square = (v_tgt ** 2).sum()
        if 0.5 ** 2 < v_tgt_square <= 0.7 ** 2:
            return 0.1
        elif v_tgt_square <= 0.5 ** 2:
            return 1.0 - 3.5 * v_tgt_square
        else:
            return 0
 

    @staticmethod
    def dense_effort_penalty(action):
        # action = 0.5 * (action + 1)  # transform action from nn to environment range
        effort = 0.5 * (action ** 2).mean()
        return -effort



    def v_tgt_deviation_penalty(self, foot_step_reward, v_body, v_tgt):
        delta_v = v_body - v_tgt
        penalty = 0.5 * np.sqrt((delta_v ** 2).sum())
        self.v_tgt_step_penalty += penalty
        if foot_step_reward != 0.0:
            step_penalty = -self.v_tgt_step_penalty / max(1, self.step_time)
            self.v_tgt_step_penalty = 0
            self.step_time = 0
            return step_penalty
        else:
            self.step_time += 1
            return 0



    def shape_reward(self, action, reward, done):

        state_desc = self.get_state_desc()

        if not self.alive_bonus: # yes.
            reward -= 0.1 * self.frame_skip

        if not self.task_bonus: # yes
            if reward >= 450:
                reward -= 500


        dead = (self.episode_len + 1) * self.frame_skip < self.timestep_limit

        if done and dead:
            reward = self.death_penalty # -50.

        v_body = self.get_v_body(state_desc)
        v_tgt = self.get_v_tgt(state_desc)
        clp = self.crossing_legs_penalty(state_desc)
        
        pvb = self.pelvis_velocity_bonus(v_tgt, v_body) #
        tab = self.target_achieve_bonus(v_tgt)
        vdp = self.v_tgt_deviation_penalty(reward, v_body, v_tgt)
        dep = self.dense_effort_penalty(action) #

        # [59] reward: 0.0, clp: 0.0, vdp: 0, pvb: 0.026, dep: -0.185, tab: 0.
        # [60] reward: 1.135, clp: 0.0, vdp: -0.973, pvb: 0.108, dep: -0.198, tab: 0.
        # [61] reward: 0.0, clp: 0.0, vdp: 0, pvb: -0.132, dep: -0.204, tab: 0.
        # [62] reward: 0.0, clp: 0.0, vdp: 0, pvb: -0.112, dep: -0.187, tab: 0.
        # [63] reward: 0.0, clp: 0.0, vdp: 0, pvb: 0.087, dep: -0.189, tab: 0.
        # [64] reward: 0.0, clp: 0.0, vdp: 0, pvb: 0.058, dep: -0.184, tab: 0.

        print(f'[{self.episode_len}] reward: {round(reward,3)}, clp: {round(clp,3)}, vdp: {round(vdp,3)}, pvb: {round(pvb,3)}, dep: {round(dep,3)}, tab: {round(tab,3)}.')
        return [reward, clp, vdp, pvb, dep, tab]



    def get_body_pos_vel(self):
        state_desc = self.get_state_desc()
        p_body = [state_desc['body_pos']['pelvis'][0], -state_desc['body_pos']['pelvis'][2]]
        v_body = [state_desc['body_vel']['pelvis'][0], -state_desc['body_vel']['pelvis'][2]]
        v_tgt = self.vtgt.get_vtgt(p_body).T
        return p_body, v_body, v_tgt

 
 


class FrameSkipWrapper(gym.Wrapper):
    def __init__(self, env, frame_skip):
        gym.Wrapper.__init__(self, env)
        self.frame_skip = frame_skip

    def reset(self):
        return self.env.reset()

    def step(self, action):
        step_reward = 0.0
        for _ in range(self.frame_skip):
            time_before_step = time.time()
            obs, reward, done, info = self.env.step(action)

            # early stopping of episode if step takes too long
            if (time.time() - time_before_step) > 10: done = True
            step_reward += reward
            if done: break
        return obs, step_reward, done, info


 

class SegmentPadWrapper(gym.Wrapper):
    def __init__(self, env, segment_len):
        gym.Wrapper.__init__(self, env)

        self.segment_len = segment_len // 2 # 5.
        self.episode_len = 0
        self.residual_len = 0

        self.episode_ended = False
        self.last_transaction = None

    def reset(self):
        self.episode_ended = False
        self.episode_len = 0
        self.residual_len = 0
        return self.env.reset()

    def check_reset(self):
        
        if (self.episode_len + self.residual_len) % self.segment_len == 0:
            self.last_transaction[3]['reset'] = True

        self.residual_len += 1


    def step(self, action):
        if self.episode_ended:
            self.check_reset()
        else:

            self.episode_len += 1

            # obs, reward, done, info
            self.last_transaction = self.env.step(action)
            self.last_transaction[3]['reset'] = False # info['reset'] = False.

            if self.last_transaction[2]: # done.
                self.episode_ended = True
                self.check_reset()

        return self.last_transaction


 




class NormalizedActions(gym.ActionWrapper): 
    # map from tanh's [-1, 1] to action_space's [low, high].
    def action(self, action):

        # print(f'## a')
        low_bound = self.action_space.low # all 0.0.
        upper_bound = self.action_space.high # all 1.0.
        
        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)

        return action

    def reverse_action(self, action):

        # print(f'## b')

        low_bound = self.action_space.low
        upper_bound = self.action_space.high

        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)

        return action



def make_env(args, seed, is_train_env=True):

    def _make_env():

        env = L2M2019Env(
            visualize=False,
            # visualize=True,
            integrator_accuracy=args.accuracy,
            difficulty=3,
            seed=seed
        )

        env = FrameSkipWrapper(env, args.frame_skip)
        env = SegmentPadWrapper(env, args.segment_len)        
        # env = NormalizedActions(env)
        env = SkeletonWrapper(env, is_train_env, args)

        return env

    return _make_env


seeds = np.random.choice(1000, size=5, replace=False)
env = make_env(train_env_args, seeds[0], True)()




 
agent = Agent()
 
obs = env.reset()  
t = 0 
while True:
    action = agent.act(obs) 

    obs, reward, done, info = env.step(action)

    # print(f"env.footstep['new']: {env.footstep['new']}")

    t += 1
    # episode_reward += reward

    # print(f'[{t}] reward: {reward}')

    if info['reset']:
        obs = env.reset() 
        t = 0


 


# # v_tgt = env.env.env.env.vtgt.get_vtgt(p_body)
# # print(f'v_tgt: {v_tgt}')


# action_space = gym.spaces.Box(low=0, high=1, shape=(22,), dtype=np.float32)

# obs = env.reset()
# raw_obs, reward, done, info = env.step(action_space.sample())

# print(f"env.footstep['new']: {env.footstep['new']}")


# print(f'# env.env.env.env.get_state_desc: {env.env.env.env.env.get_state_desc}')
# print(f'# env.get_state_desc: {env.get_state_desc}')
# print(f'# env.env.env.env.env.get_state_desc==env.get_state_desc: {env.env.env.env.env.get_state_desc==env.get_state_desc}')


# print(f'# env.env.env.env.vtgt.get_vtgt: {env.env.env.env.env.vtgt.get_vtgt}')
# print(f'# env.vtgt.get_vtgt: {env.vtgt.get_vtgt}')
# print(f'# env.env.env.env.vtgt.get_vtgt == env.get_vtgt: {env.env.env.env.env.vtgt.get_vtgt == env.vtgt.get_vtgt}')

 
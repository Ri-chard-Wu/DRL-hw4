
from time import time
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

        if(self.i % 4 == 0):            
            self.i = 1

            obs_vec = obs2vec(observation) 
            obs_vec = tf.convert_to_tensor(obs_vec[None, None, ...], dtype=tf.float32)
           
            # action = self._act(obs_vec, False)[0, 0].numpy() # (22,)            

            mean, _ = self.policy_net(obs_vec, training=False) # (b, T+1, action_dim).
            action = tf.math.tanh(mean)[0, 0].numpy()
            

            action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
            action = np.clip(action, low_bound, upper_bound)
            
            self.prev_action = action
  
        else:
            self.i += 1 
            
        return self.prev_action  

 

  
    def load(self, path):
         
        with open(path, 'rb') as f:            
            state_dict = pickle.load(f)                    
 
        self.policy_net.set_weights(state_dict['policy_net'])
 
 
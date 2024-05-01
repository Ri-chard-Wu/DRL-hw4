
 
import numpy as np
  
import tensorflow.keras.backend as K
import tensorflow as tf
 
from importlib import import_module

from env_wrappers import *
 
# from train import learn



# from osim.env import L2M2019Env
 
# env = L2M2019Env(visualize=False) 
# print(f'env.observation_space: {env.observation_space}')
# print(f'env.action_space: {env.action_space}')
# print(f"getattr(env, 'v_tgt_field_size', 0): {getattr(env, 'v_tgt_field_size', 0)}")
# print(f'env.spec: {env.spec}')
 

# exit()
 

class AttrDict(dict):
    def __getattr__(self, a):        
        return self[a]
  

args = AttrDict({
    
    'k': 4,   
    'n_env': 4,
    'max_episode_length': 625,


    'batch_size': 256,
    
    'memory_size': int(1e6),
    'n_prefill_steps': 1000,

    'seed': 1,
})

 

alg_args = {'policy_hidden_sizes': (256, 256),
            'value_hidden_sizes': (256, 256),
            'q_hidden_sizes': (256, 256),
            'discount': 0.96,
            'tau': 0.01,
            'lr': 1e-3,
            'policy_lr': 1e-2,
            'sym_memory': False,                        
            'alpha': 0.2,
            'learn_alpha': True, 
            'loss_ord': 1,
            'n_sample_actions': 10}


expl_args = {'n_state_predictors': 5,
            'state_predictor_hidden_sizes': (64, 64),
            'lr': 1e-3,
            'bonus_scale': 1}




def build_env():


    def make_env():
        
        env_args = {'model': '3D', 'visualize': False, 'integrator_accuracy': 1e-3, 'difficulty': 3, 'stepsize': 0.01}

        env = L2M2019EnvBaseWrapper(**env_args)
        env = RandomPoseInitEnv(env)
        env = ActionAugEnv(env) # transform action from tanh policies in (-1,1) to (0,1) before passing down.
        env = PoolVTgtEnv(env, **env_args)
        env = RewardAugEnv(env)
        env = SkipEnv(env)
        env = Obs2VecEnv(env)
 

        print(f'# env.time_limit: {env.time_limit}, env.n_skips: {env.n_skips}')
    
        return env

 
    return VecEnv([make_env for i in range(args.n_env)])
   


 


env = build_env() 


np.random.seed(args.seed)
tf.set_random_seed(args.seed) 

agent = learn(env, args, expl_args, alg_args)




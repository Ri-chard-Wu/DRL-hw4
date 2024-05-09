
import numpy as np

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





observation_shape = [2 * 11 * 11, 97]
action_shape = 22
low_bound = np.zeros((action_shape,))
upper_bound = np.ones((action_shape,))   


train_env_args = AttrDict({
            "env_num": 16,
            "segment_len": 10,

            "difficulty": 3,
            "accuracy": 1e-4,
            "frame_skip": 4,
            "timestep_limit": 10000,
    
            'alive_bonus': 0,
            'death_penalty': -50.0,
            'task_bonus': 1,
        })




 


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
 
 

critic_args = AttrDict({
    "hidden_dim": 1024,
    "noisy": "False",
    "layer_norm": True,
    "afn": "relu",
    "residual": True,
    "dropout": 0.1,
    "q_value_dim": 6,
    "lr": 1e-4
})




sac_args = AttrDict({
    "gamma": 0.99,
    "soft_tau": 1e-2,

    "n_step_loss": 5,

    "rescaling": True,
    
    "n_step_train": 10,

    "priority_weight": 0.9,
    "q_weights": [2.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  })
   


trainer_args = AttrDict({

    "start_priority_exponent": [0.2, 0.89][1],
    "end_priority_exponent": 0.9,

    "start_importance_exponent": [0.2, 0.89][1],
    "end_importance_exponent": 0.9,
    
    "prioritization_steps": 3000,
    "exp_replay_checkpoint": None,

 
    "log_interval": 5,
    "save_interval": [1, 100][1],
    "save_exp_interval": [1, 500][1],

    "save_dir": "ckpt",    
    "load_ckpt": ["ckpt/ckpt-300.h5", "ckpt/best.h5"][0],
    "load_exp": "ckpt/exp.h5",

    "min_experience_len": [0, 2, 50][0],
    
    "epoch_size": 100000000, #500,
    "batch_size": 256,
    "train_steps": 16,

    "test_n": 5,

    "render": False,
    "segment_file": None,
    "pretrain_critic": False,
    "num_pretrain_epoch": 0  
})
  

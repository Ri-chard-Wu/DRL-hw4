
from utils import AttrDict
 




observation_shape = [2 * 11 * 11, 97]
action_shape = 22




trainer_args = AttrDict({

    "start_priority_exponent": 0.2,
    "end_priority_exponent": 0.9,

    "start_importance_exponent": 0.2,
    "end_importance_exponent": 0.9,
    
    "prioritization_steps": 3000,
    "exp_replay_checkpoint": None,

    # "agent_checkpoint": "logs/learning_to_move/8c/epoch_0.pth",
    # "load_full": "True"



    "min_experience_len": 100,
    "num_epochs": 40,
    "epoch_size": 500,
    "batch_size": 256,
    "train_steps": 16,
    "test_n": 3,
    "render": False,
    "segment_file": None,
    "pretrain_critic": False,
    "num_pretrain_epoch": 0  
})
  

 
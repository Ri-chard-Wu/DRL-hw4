import gym
import json
import torch

from env_wrappers import make_train_env, make_test_env

from src.nn import create_nets_sac, create_optimizers_sac
from src.losses import QValueLoss, NStepQValueLoss, NStepQValueLossSeparateEntropy
from src.sac import SAC
from src.r2d2.segment_sampler import SegmentSampler
from src.r2d2.experience_replay import PrioritizedExperienceReplay, ExperienceReplay
from src.r2d2.writer import Writer
from src.r2d2.trainer import Trainer
 
from utils import AttrDict


 

env_num = parameters['environment']['env_num']
segment_len = parameters['environment']['segment_len']

 
train_env = make_train_env()
test_env make_test_env()

observation_space = [2 * 11 * 11, 97]
action_space = 22


q_value_dim = 6

policy_net, q_net_1, q_net_2, target_q_net_1, target_q_net_2, policy_optim, q1_optim, q2_optim = \
                            create_nets(observation_space, action_space)

 


# ================================= init agent =================================
agent_parameters = parameters['agent_parameters']
gamma = agent_parameters['gamma']
soft_tau = agent_parameters['soft_tau']
n_step_loss = agent_parameters['n_step_loss']
rescaling = agent_parameters['rescaling'] == 'True'
q_weights = agent_parameters['q_weights']

q_value_loss = NStepQValueLossSeparateEntropy(gamma, device, q_weights, n_steps=n_step_loss, rescaling=rescaling)

n_steps = agent_parameters['n_step_train']  # number of steps from tail of segment to learn from
# aka eta, priority = eta * max_t(delta) + (1 - eta) * mean_t(delta)
priority_weight = agent_parameters['priority_weight']
use_observation_normalization = agent_parameters['use_observation_normalization'] == 'True'

agent = SAC(
    policy_net, q_net_1, q_net_2, target_q_net_1, target_q_net_2,
    q_value_loss,
    policy_optim, q1_optim, q2_optim,
    soft_tau, device, action_space, n_steps,
    priority_weight,
    q_value_dim, q_weights,
    use_observation_normalization
)

# ================== init segment sampler & experience replay ==================
replay_parameters = parameters['replay_parameters']
log_dir = replay_parameters['log_dir']

segment_sampler = SegmentSampler(agent, train_env, segment_len, q_weights)

replay_capacity = replay_parameters['replay_capacity']  # R2D2 -> 100k
actor_size = replay_parameters['actor_size']
critic_size = replay_parameters['critic_size']
prioritization = replay_parameters['prioritization'] == 'True'
exp_replay_init_fn = PrioritizedExperienceReplay if prioritization else ExperienceReplay
experience_replay = exp_replay_init_fn(
    replay_capacity, # 0.25e6.
    segment_len, # 10.
    observation_space, # [2 * 11 * 11, 97].
    action_space, # 22.
    q_value_dim, # 6.
    # (1, 2) for feed forward net, (hidden_size, hidden_size * 2) for lstm of mhsa
    actor_size, # 1.
    critic_size # 2.
)

# ================================ init trainer ================================
trainer_parameters = parameters['trainer_parameters']
start_priority_exponent = trainer_parameters['start_priority_exponent']
end_priority_exponent = trainer_parameters['end_priority_exponent']
start_importance_exponent = trainer_parameters['start_importance_exponent']
end_importance_exponent = trainer_parameters['end_importance_exponent']
prioritization_steps = trainer_parameters['prioritization_steps']

exp_replay_checkpoint = trainer_parameters['exp_replay_checkpoint']
if exp_replay_checkpoint == 'None':
    exp_replay_checkpoint = None
agent_checkpoint = trainer_parameters['agent_checkpoint']
if agent_checkpoint == 'None':
    agent_checkpoint = None
load_full = trainer_parameters['load_full'] == 'True'

trainer = Trainer(
    env_num, test_env,
    segment_sampler, log_dir,
    agent, 
    experience_replay,
    start_priority_exponent, 
    end_priority_exponent, # 0.9.
    start_importance_exponent, end_importance_exponent,
    q_value_dim
)
trainer.load_checkpoint(agent_checkpoint, load_full)

# ================================ train agent  ================================
training_parameters = parameters['training_parameters']
min_experience_len = training_parameters['min_experience_len']
num_epochs = training_parameters['num_epochs']
epoch_size = training_parameters['epoch_size']
batch_size = training_parameters['batch_size']
train_steps = training_parameters['train_steps']
test_n = training_parameters['test_n']
render = training_parameters['render'] == 'True'

segment_file = training_parameters['segment_file']
pretrain_critic = training_parameters['pretrain_critic'] == 'True'
num_pretrain_epoch = training_parameters['num_pretrain_epoch']



if num_pretrain_epoch > 0: # no.
    trainer.pretrain_from_segments(
        segment_file, num_pretrain_epoch, batch_size,
        actor_size, critic_size
    )


trainer.train(
    min_experience_len,
    num_epochs, epoch_size,
    train_steps, batch_size,
    test_n, render,
    prioritization_steps,
    pretrain_critic,
    exp_replay_checkpoint
)

train_env.close()
test_env.close()



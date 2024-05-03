
 
import numpy as np
   
import tensorflow as tf
 
from memory import Memory
from explore import DisagreementExploration
from model import SAC

from importlib import import_module

from env_wrappers import *
 
 

# from osim.env import L2M2019Env
 
# env = L2M2019Env(visualize=False) 
# print(f'env.observation_space: {env.observation_space.shape}')
# print(f'env.action_space: {env.action_space.shape}')
# print(f"getattr(env, 'v_tgt_field_size', 0): {getattr(env, 'v_tgt_field_size', 0)}")
# print(f'env.spec: {env.spec}')
 


# exit()
 

class AttrDict(dict):
    def __getattr__(self, a):        
        return self[a]
  
 

args = AttrDict({
    
    'k': 4,   
    'n_env': 8,
    'max_episode_length': 625,


    'batch_size': 256,
    
    'memory_size': int(1e6),
    'n_prefill_steps': 80,

    'seed': 1,

    'log_interval': 10
})

 

alg_args = AttrDict({'policy_hidden_sizes': (256, 256),
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
            'n_sample_actions': 10})


expl_args = AttrDict({'n_state_predictors': 5,
            'state_predictor_hidden_sizes': (64, 64),
            'lr': 1e-3,
            'bonus_scale': 1})




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



 

# # obs = env.reset()
# # obs_next, r, done, info = env.step(np.random.rand(4, 22))
# # # print(f'r.shape: {r.shape}, done.shape: {done.shape}')
# # # a = np.vstack([i.get('rewards', 0) for i in info])
# # print(f'done: {done}')
# print(f'env.observation_space.shape: {env.observation_space.shape}')
# print(f'env.action_space.shape: {env.action_space.shape}')
# # # # obs = env.reset()
# # # # print(f'obs.shape: {obs.shape}')
# exit()


 
np.random.seed(args.seed)
# tf.set_random_seed(args.seed) 
tf.random.set_seed(args.seed)

episode_lengths = np.zeros((env.num_envs, 1), dtype=int)


# env.observation_space.shape: (102,),  env.action_space.shape: (22,).
memory = Memory(args.memory_size, env.observation_space.shape, env.action_space.shape)
agent = SAC(env.observation_space.shape, env.action_space.shape, env.v_tgt_field_size, alg_args)
exploration = DisagreementExploration(env.observation_space.shape, env.action_space.shape, expl_args)


memory.prefill(env, args.n_prefill_steps // env.num_envs)

obs = env.reset() # obs.shape: (n_env, 102).


# with open("eval.txt", "w") as f: f.write("")
# with open("log.txt", "w") as f: f.write("")

for t in range(int(1e8)):
    
    act = agent.get_actions(obs)  # (n_samples, n_env, action_dim=22)
    act = exploration.select_best_action(obs, act) # (n_env, action_dim=22)
    obs_next, rew, done, info = env.step(act) # r: (n_env, 1), done: (n_env, 1)
    r_aug = np.vstack([i.get('rewards', 0) for i in info]) # (n_env, 1)
    r_bonus = exploration.get_exploration_bonus(obs, act) # encourage exploration.
    
    # don't count "reaching max_episode_length (time limit)" as done.
    done_bool = np.where(episode_lengths + 1 == args.max_episode_length, np.zeros_like(done), done)  

    memory.store_transition(obs, act, rew + r_bonus + r_aug, done_bool, obs_next)
    obs = obs_next

    episode_lengths += 1

    # end of episode -- when all envs are done or max_episode length is reached, reset
    if any(done):
        # set episode_lengths of those whose done is True to 0.
        for d in np.nonzero(done)[0]:                                     
            episode_lengths[d] = 0



    batch = memory.sample(args.batch_size)
    policy_loss, v_loss, q_loss, alpha_loss = agent.train_step(batch)
    
    expl_loss = exploration.train(memory, args.batch_size)

 
    # print(f't: {t}')

    if(t%args.log_interval==0):
        log = {}
        log['policy_loss'] = policy_loss.round(4)
        log['v_loss'] = v_loss.round(4)
        log['q_loss'] = q_loss.round(4)
        log['alpha_loss'] = alpha_loss.round(4)
        log['expl_loss'] = expl_loss.round(4)
        s = f't: {t}, ' + str(log)
        with open("log.txt", "a") as f: f.write(s + '\n')
        print(s)









    # if args.play:
    #     if env_args: env_args['visualize'] = True
    #     env = make_single_env(args.env, args.rank, args.n_env + 100, args.seed, env_args, args.output_dir)
    #     obs = env.reset()
    #     episode_rewards = 0
    #     episode_steps = 0

    #     while True: 
    #         action = agent.get_actions(obs)  # (n_samples, batch_size, action_dim)
    #         action = exploration.select_best_action(np.atleast_2d(obs), action)
    #         next_obs, rew, done, info = env.step(action.flatten())
    #         r_bonus = exploration.get_exploration_bonus(np.atleast_2d(obs), action, np.atleast_2d(next_obs)).squeeze()
    #         episode_rewards += rew
    #         episode_steps += 1
 
    #         obs = next_obs
    #         env.render()
    #         if done:
    #             print('Episode length {}; cumulative reward: {:.2f}'.format(episode_steps, episode_rewards))
    #             episode_rewards = 0
    #             episode_steps = 0
    #             i = input('enter random seed: ')
    #             obs = env.reset(seed=int(i) if i is not '' else None)

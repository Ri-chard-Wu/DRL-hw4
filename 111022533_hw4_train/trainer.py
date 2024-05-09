 
import pickle
import numpy as np
from tqdm import trange
import time
from sac import SAC

import h5py

import tensorflow.keras.backend as K
import tensorflow as tf

import gym
from osim.env import L2M2019Env

import os
from parameters import trainer_args as args
from parameters import train_env_args


 
class Trainer:
 
    def __init__(self, segment_sampler, agent, experience_replay):

      
        self.segment_sampler = segment_sampler
 

        self.experience_replay = experience_replay

        self.priority_exponent = args.start_priority_exponent # 0.2.
        self.end_priority_exponent = args.end_priority_exponent # 0.9.
        self.importance_exponent = args.start_importance_exponent
        self.end_importance_exponent = args.end_importance_exponent


        self.agent = agent 

        if("load_ckpt" in args):
            self.agent.load(args.load_ckpt)
 

        if("load_exp" in args):
            self.load_exp_replay(args.load_exp)



    def save_exp_replay(self, dir_name, name):

        print('saving exp_replay...')
 
        path = os.path.join(dir_name, name)


        f = h5py.File(path, mode='w')
        data_group = f.create_group('experience_replay')
        # data_group.create_dataset('capacity', data=self.experience_replay.capacity)

        for k, v in self.experience_replay.tree.__dict__.items():
            
            if hasattr(v, '__len__'):
                data_group.create_dataset(k, data=v, compression="lzf")  # can compress only array-like structures
            else:
                data_group.create_dataset(k, data=v)  # can't compress scalars

        f.close()
    

        
    

    def load_exp_replay(self, path):

        print('loading exp replay...')
  
        # self.experience_replay = self.load_experience_replay_from_h5(self.experience_replay, filename)
        
        f = h5py.File(path, mode='r')
        data_group = f['experience_replay']

        for key in self.experience_replay.tree.__dict__:
            loaded_value = data_group[key][()]
            self.experience_replay.tree.__dict__.update({key: loaded_value})

        f.close()




   
    def eval(self, ckpt_dir, ckpt_name):

        print(f'evaluating {ckpt_name} ...')
        
        path = os.path.join(ckpt_dir, ckpt_name)

        agent = SAC()
        agent.load(path)
 
        total_reward = 0
        time_limit = 120

        env = L2M2019Env(visualize=False, difficulty=2)

        for episode in range(10):

            obs = env.reset()
            start_time = time.time()
            episode_reward = 0
            
            t = 0
            while True:
                action = agent.act(obs) 

                obs, reward, done, info = env.step(action)
                t += 1
                episode_reward += reward

                if time.time() - start_time > time_limit:
                    print(f"Time limit reached for episode {episode}")
                    break

                if done or t >= env.spec.timestep_limit-1:
                    break
 
            total_reward += episode_reward
            
            print(f'episode-{episode} reward: {episode_reward}')

        env.close()

        score = total_reward / 10
        print(f"Final Score: {score}")

        return score





    def sample_new_experience(self):

        # new_segment.segment - obs: (n_env, T+1, dim), act, rew: (n_env, T, dim), don: (n_env, T). T==10.
        # priority: (n_env,)        
        new_segment, priority = self.segment_sampler.sample()

        self.experience_replay.push(new_segment, priority, self.priority_exponent)



    def train_step(self):


        segment, exp_ids, importance_weights = \
                        self.experience_replay.sample(args.batch_size, self.importance_exponent)

        losses, q_min, upd_priority = self.agent.train_step(\
                        segment, importance_weights) # (5,), (q_dim,), (b,)

        K.clear_session()
        tf.keras.backend.clear_session() 

        self.experience_replay.update_priorities(exp_ids, upd_priority, self.priority_exponent)

        return losses, q_min # (5,), (q_dim,)

 
         
 

    def train(self):
        '''
        batch size here is the number of segments to sample from experience replay
        '''

        # end_priority_exponent: 0.9.
        # priority_exponent: 0.2.
        priority_delta = (self.end_priority_exponent - self.priority_exponent) / args.prioritization_steps
        importance_delta = (self.end_importance_exponent - self.importance_exponent) / args.prioritization_steps

        self.segment_sampler.sample_first_half_segment()

        
        for _ in trange(args.min_experience_len):             
            self.sample_new_experience() # will push to replay buffer.

  
 

     
        for t in range(args.epoch_size): 
 
            self.sample_new_experience() # will push to replay buffer.

            for _ in range(args.train_steps):

                segment, exp_ids, importance_weights = \
                                self.experience_replay.sample(args.batch_size, self.importance_exponent)

                losses, upd_priority = self.agent.train_step(segment, importance_weights) # (4,), (b,)

                K.clear_session()
                tf.keras.backend.clear_session() 

                self.experience_replay.update_priorities(exp_ids, upd_priority, self.priority_exponent)

 

            self.importance_exponent += importance_delta
            self.importance_exponent = min(self.end_importance_exponent, self.importance_exponent)

            self.priority_exponent += priority_delta
            self.priority_exponent = min(self.end_priority_exponent, self.priority_exponent)



            if(t%args.log_interval==0):
                
                print(f'[{t}] losses: {losses.round(3)}')
                with open("train.txt", "a") as f: f.write(f'[{t}] losses: {losses.round(3)}' + '\n')



            if(t%args.save_interval==0):

                self.agent.save(args.save_dir, f'ckpt-{t}.h5')

            
                mean_reward = self.eval(args.save_dir, f'ckpt-{t}.h5')
                with open("eval.txt", "a") as f: 
                    f.write(f'[{t}] mean_reward: {round(mean_reward, 3)}' + '\n')
             



            if(t%args.save_exp_interval==0):
                self.save_exp_replay(args.save_dir, 'exp.h5')
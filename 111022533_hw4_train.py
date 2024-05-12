 
import numpy as np

import time
from sac import SAC 

import tensorflow.keras.backend as K
import tensorflow as tf

import gym
from osim.env import L2M2019Env

import os

from parameters import trainer_args as args
from parameters import train_env_args
 
from env_wrappers import make_train_env

  
from replay_buffer import PrioritizedExperienceReplay
  













class Trainer:
 
    def __init__(self, ):

 
        self.replay_buffer = PrioritizedExperienceReplay()

        self.priority_exponent = args.start_priority_exponent # 0.2.
        self.end_priority_exponent = args.end_priority_exponent # 0.9.

        self.importance_exponent = args.start_importance_exponent
        self.end_importance_exponent = args.end_importance_exponent

 
        self.agent = SAC()  

        if("load_ckpt" in args):
            self.agent.load(args.load_ckpt)
 
        if("load_exp" in args): 
            self.replay_buffer.load_data(args.load_exp)


        self.env = make_train_env()  
        self.cur_half_seg, self.cur_obs = self.sample_half_segment(self.env.reset())  

 

    def sample_half_segment(self, obs):
 
        obs_seg = []
        act_seg = []
        rew_seg = []
        don_seg = []

        for _ in range(train_env_args.segment_len // 2): # 10 // 2 == 5.

            obs_seg.append(obs)
 
            act = np.squeeze(self.agent.act_sampler(obs[:,None,:]), axis=1)                
            obs, rew, don, _ = self.env.step(act)
                    
            act_seg.append(act)
            rew_seg.append(rew) # (n_env, 6). 
            don_seg.append(don)
         
        orders = [(1, 0, 2), (1, 0, 2), (1, 0, 2), (1, 0)]


        # print()

        seg = map(np.array, (obs_seg, act_seg, rew_seg, don_seg))        
        seg = [s.transpose(order) for s, order in zip(seg, orders)]

        return seg, obs 




    def sample_new_experience(self):
 

        next_half_seg, new_observation = self.sample_half_segment(self.cur_obs)
  
        # obs, act, rew: (n_env, T, dim), don: (n_env, T). T==10.
        seg = [np.concatenate((s1, s2), axis=1) for s1, s2 \
                                    in zip(self.cur_half_seg, next_half_seg)]

        seg[0] = np.concatenate((seg[0], new_observation[:, None, :]), 1) # (n_env, T+1, dim)

        # segment - obs: (n_env, T+1, dim). act, rew: (n_env, T, dim). don: (n_env, T). T==10.
        priority = self.agent.calculate_priority_loss(seg) # (n_env,).
 
     

        self.cur_half_seg, self.cur_obs = next_half_seg, new_observation         
        # if(any(list(next_half_seg)[3])):
        #     self.cur_half_seg, self.cur_obs = self.sample_half_segment(new_observation)  

        
        self.replay_buffer.push(seg, priority, self.priority_exponent)



    def train_step(self):


        segment, exp_ids, importance_weights = \
                        self.replay_buffer.sample(args.batch_size, self.importance_exponent)

        losses, q_min, upd_priority = self.agent.train_step(\
                        segment, importance_weights) # (5,), (q_dim,), (b,)

     
        self.replay_buffer.update_priorities(exp_ids, upd_priority, self.priority_exponent)

        return losses, q_min # (5,), (q_dim,)

 
         
 

    def train(self):
        '''
        batch size here is the number of segments to sample from experience replay
        '''

        priority_delta = (self.end_priority_exponent - self.priority_exponent) / args.prioritization_steps
        importance_delta = (self.end_importance_exponent - self.importance_exponent) / args.prioritization_steps
 
        
        for _ in range(args.min_experience_len):
            self.sample_new_experience() # will push to replay buffer.

   
        for t in range(args.epoch_size): 
 
            self.sample_new_experience() # will push to replay buffer.

            for _ in range(args.train_steps):

                segment, exp_ids, importance_weights = \
                                self.replay_buffer.sample(args.batch_size, self.importance_exponent)

                losses, upd_priority = self.agent.train_step(segment, importance_weights) # (4,), (b,)

                K.clear_session()
                tf.keras.backend.clear_session() 

                self.replay_buffer.update_priorities(exp_ids, upd_priority, self.priority_exponent)

 

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
                self.replay_buffer.save_data(args.save_dir, 'exp.h5')




   
    def eval(self, ckpt_dir, ckpt_name):

        print(f'evaluating {ckpt_name} ...')
        
        path = os.path.join(ckpt_dir, ckpt_name)

        agent = SAC()
        agent.load(path)
 
        total_reward = 0
        time_limit = 120

        env = L2M2019Env(visualize=False, difficulty=2)

        n = 6

        for episode in range(n):

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

        score = total_reward / n
        print(f"Final Score: {score}")

        return score




            
trainer = Trainer()
trainer.train()

 
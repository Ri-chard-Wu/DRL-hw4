import numpy as np  
  
from parameters import train_env_args
from env_wrappers import make_train_env




class SegmentSampler:
    
    def __init__(self, agent):

        self.agent = agent
  
        self.env = make_train_env() 
        obs = self.env.reset()
        
        self.cur_half_seg, self.cur_obs = self.sample_half_segment(obs)       


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

        seg = map(np.array, (obs_seg, act_seg, rew_seg, don_seg))        
        seg = [s.transpose(order) for s, order in zip(seg, orders)]

        return seg, obs 




    def sample(self):
 

        next_half_seg, new_observation = self.sample_half_segment(self.cur_obs)
  
        # obs, act, rew: (n_env, T, dim), don: (n_env, T). T==10.
        segment = np.concatenate((self.cur_half_seg, next_half_seg), axis=1)
        segment[0] = np.concatenate((segment[0], new_observation[:, None, :]), 1) # (n_env, T+1, dim)

        # segment - obs: (n_env, T+1, dim). act, rew: (n_env, T, dim). don: (n_env, T). T==10.
        priority_loss = self.agent.calculate_priority_loss(segment) # (n_env,).
 
     
        self.cur_half_seg = next_half_seg
        self.cur_obs = new_observation

        return segment, priority_loss  

 
import pickle
import numpy as np
from tqdm import trange

import h5py

from parameters import trainer_args as args
from parameters import train_env_args


 
class Trainer:
 
    def __init__(self, test_env, segment_sampler, agent, experience_replay):

        self.test_env = test_env
        self.segment_sampler = segment_sampler
        # self.logdir = logdir
        
        self.agent = agent 
        self.experience_replay = experience_replay

        self.priority_exponent = args.start_priority_exponent # 0.2.
        self.end_priority_exponent = args.end_priority_exponent # 0.9.
        self.importance_exponent = args.start_importance_exponent
        self.end_importance_exponent = args.end_importance_exponent


 

    # def save_exp_replay(self, epoch):
    #     print('saving exp_replay...')

    #     self.experience_replay
       

    # def load_exp_replay(self, filename):

    #     print('loading exp replay...')
    #     filename = str(filename)

    #     if filename.endswith('.pickle'):

    #         with open(filename, 'rb') as f:
    #             self.experience_replay = pickle.load(f)

    #     elif filename.endswith('.h5'):
    #         self.experience_replay = self.load_experience_replay_from_h5(self.experience_replay, filename)
    #     else:
    #         raise ValueError("don't know ho to parse this type of file")




    def _test_agent(self): 
        # tests only non-shaped reward
        episode_reward = 0.0
        observation, done = self.test_env.reset(), False
        
        if args.render:
            self.test_env.render()

        while not done:
            action = self.agent.act_test(observation)
            observation, reward, done, _ = self.test_env.step(action)

            if args.render:
                self.test_env.render()

            episode_reward += reward 
        return episode_reward

 
    def test_n(self):
        
        print('testing agent...')
 
        mean_total_reward = 0.0
        
        for i in range(args.test_n):
            episode_reward = self._test_agent()
            print('episode {} reward: {}'.format(i, episode_reward))
            mean_total_reward += episode_reward
           
        mean_total_reward /= args.test_n
    
    
        return mean_total_reward



    def sample_new_experience(self):

        # new_segment.segment - obs: (n_env, T+1, dim), act, rew: (n_env, T, dim), don: (n_env, T). T==10.
        # priority: (n_env,)        
        new_segment, priority = self.segment_sampler.sample()

        self.experience_replay.push(new_segment, priority, self.priority_exponent)



    def _train_step(self, batch_size,
                     learn_policy # True.
                     ):

        # sample a batch of segments of data.
        # segment_data, exp_ids, importance_weights = self.experience_replay.sample(batch_size, self.importance_exponent)

        segment, exp_ids, importance_weights = \
                        self.experience_replay.sample(batch_size, self.importance_exponent)

        losses, q_min, upd_priority = self.agent.learn_from_data(\
                        segment, importance_weights, learn_policy) # (5,), (q_dim,), (b,)


        self.experience_replay.update_priorities(
            exp_ids, upd_priority, self.priority_exponent
        )

        return losses, q_min # (5,), (q_dim,)

 
        
    def _train_epoch(self, epoch, importance_delta, priority_delta, learn_policy=True):        

      
        for _ in trange(args.epoch_size, desc=f'epoch_{epoch}'): 
 
            self.sample_new_experience() # will push to replay buffer.

            for train_step in range(args.train_steps):
                self._train_step(args.batch_size, learn_policy) # (5,), (q_dim,)
        
            self.importance_exponent += importance_delta
            self.importance_exponent = min(self.end_importance_exponent, self.importance_exponent)

            self.priority_exponent += priority_delta
            self.priority_exponent = min(self.end_priority_exponent, self.priority_exponent)



    def train(self):
        '''
        batch size here is the number of segments to sample from experience replay
        '''

        # end_priority_exponent: 0.9.
        # priority_exponent: 0.2.
        priority_delta = (self.end_priority_exponent - self.priority_exponent) / args.prioritization_steps
        importance_delta = (self.end_importance_exponent - self.importance_exponent) / args.prioritization_steps

        self.segment_sampler.sample_first_half_segment()

        
        # for _ in trange(args.min_experience_len): # 100.
        for _ in trange(2): # 100.
            self.sample_new_experience() # will push to replay buffer.

        # self.save_exp_replay(0)
 

        for epoch in range(args.num_epochs):

            self._train_epoch(epoch, importance_delta, priority_delta)

            self.save_exp_replay(epoch + 1)  

            # TODO: save model & relay buffer and do eval.

            # test_reward = self.test_n()

            

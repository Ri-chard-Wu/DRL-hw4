 
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
        if("load_ckpt" in args):
            self.agent.load(args.load_ckpt)

        self.experience_replay = experience_replay

        self.priority_exponent = args.start_priority_exponent # 0.2.
        self.end_priority_exponent = args.end_priority_exponent # 0.9.
        self.importance_exponent = args.start_importance_exponent
        self.end_importance_exponent = args.end_importance_exponent


 

 
    def load_experience_replay_from_h5(er, filename):
        # update experience replay in-place
        f = h5py.File(filename, mode='r')
        data_group = f['experience_replay']

        for key in er.tree.__dict__:
            loaded_value = data_group[key][()]
            er.tree.__dict__.update({key: loaded_value})

        f.close()
        return er
 

    def save_experience_replay_as_h5(er, filename):

        f = h5py.File(filename, mode='w')
        data_group = f.create_group('experience_replay')
        data_group.create_dataset('capacity', data=er.capacity)

        for k, v in er.tree.__dict__.items():
            
            if hasattr(v, '__len__'):
                data_group.create_dataset(k, data=v, compression="lzf")  # can compress only array-like structures
            else:
                data_group.create_dataset(k, data=v)  # can't compress scalars

        f.close()
        return



    def save_exp_replay(self, epoch):

        print('saving exp_replay...')
        # self.save_experience_replay_as_h5(self.experience_replay, self.logdir + 'exp_replay_{}.h5'.format(epoch))
    

    def load_exp_replay(self, filename):

        print('loading exp replay...')
        filename = str(filename)

        if filename.endswith('.pickle'):

            with open(filename, 'rb') as f:
                self.experience_replay = pickle.load(f)

        elif filename.endswith('.h5'):
            self.experience_replay = self.load_experience_replay_from_h5(self.experience_replay, filename)
        else:
            raise ValueError("don't know ho to parse this type of file")








    def _test_agent(self): 
        # tests only non-shaped reward
        episode_reward = 0.0
        observation, done = self.test_env.reset(), False
         
        while not done:
            action = self.agent.act_test(observation)
            observation, reward, done, _ = self.test_env.step(action)
 
            episode_reward += reward 

        return episode_reward

 
    def test_n(self):
        
        print('evaluating...')
 
        mean_reward = 0.0
        
        for i in range(args.test_n):
 
            episode_reward = 0.0
            observation, done = self.test_env.reset(), False
            
            while not done:
                action = self.agent.act_test(observation)
                observation, reward, done, _ = self.test_env.step(action)
    
                episode_reward += reward             
            

            print(f'episode {i} reward: {episode_reward}')
            mean_reward += episode_reward
           
        mean_reward /= args.test_n

        print(f'mean_reward: {mean_reward}')
    
        return mean_reward



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

      
        for t in range(args.epoch_size): 
 
            self.sample_new_experience() # will push to replay buffer.

            for train_step in range(args.train_steps):
                losses, _ = self._train_step(args.batch_size, learn_policy) # (5,), (q_dim,)
        
            self.importance_exponent += importance_delta
            self.importance_exponent = min(self.end_importance_exponent, self.importance_exponent)

            self.priority_exponent += priority_delta
            self.priority_exponent = min(self.end_priority_exponent, self.priority_exponent)



            if(t%args.log_interval==0):
                
                print(f'[{t}] losses: {losses.round(3)}')

                # log = {}
                # log['policy_loss'] = policy_loss.round(4)
                # log['v_loss'] = v_loss.round(4)
                # log['q_loss'] = q_loss.round(4)
                # log['alpha_loss'] = alpha_loss.round(4)
                # log['expl_loss'] = expl_loss.round(4)

                # s = f't: {t}, ' + str(log)

                with open("train.txt", "a") as f: f.write(f'[{t}] losses: {losses.round(3)}' + '\n')



            if(t%args.save_interval==0):
                self.agent.save(args.save_dir, f'ckpt-{t}.h5')

                mean_reward = self.test_n()
                with open("eval.txt", "a") as f: 
                    f.write(f'[{t}] mean_reward: {mean_reward.round(3)}' + '\n')


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
            # for _ in trange(50):  
            self.sample_new_experience() # will push to replay buffer.

        # self.save_exp_replay(0)
 

        for epoch in range(args.num_epochs):

            self._train_epoch(epoch, importance_delta, priority_delta)

            # self.save_exp_replay(epoch + 1)  

            # TODO: save model & relay buffer and do eval.

            # test_reward = self.test_n()

            

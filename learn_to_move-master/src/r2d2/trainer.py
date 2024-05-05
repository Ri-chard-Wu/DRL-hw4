import torch
import pickle
import numpy as np
from tqdm import trange

import h5py


class Trainer:
    def __init__(self,
                 env_num, test_env, segment_sampler,
                 logdir, agent, experience_replay,
                 start_priority_exponent, end_priority_exponent,
                 start_importance_exponent, end_importance_exponent,
                 q_dim):
        self.test_env = test_env
        self.segment_sampler = segment_sampler
        self.logdir = logdir
        
        self.agent = agent
        self.q_dim = q_dim
        self.experience_replay = experience_replay

        self.priority_exponent = start_priority_exponent # 0.2.
        self.end_priority_exponent = end_priority_exponent # 0.9.
        self.importance_exponent = start_importance_exponent
        self.end_importance_exponent = end_importance_exponent

        self.best_reward = -float('inf')
        self.collected_experience = 0
        self.exp_replay_save_frequency = env_num * self.experience_replay.capacity // 2

    @staticmethod
    def load_experience_replay_from_h5(er, filename):
        # update experience replay in-place
        f = h5py.File(filename, mode='r')
        data_group = f['experience_replay']
        for key in er.tree.__dict__:
            loaded_value = data_group[key][()]
            er.tree.__dict__.update({key: loaded_value})
        f.close()
        return er

    @staticmethod
    def save_experience_replay_as_h5(er, filename):
        f = h5py.File(filename, mode='w')
        data_group = f.create_group('experience_replay')
        data_group.create_dataset('capacity', data=er.capacity)
        for k, v in er.tree.__dict__.items():
            # example keys: _capacity, _index, _full, _sum_tree, _observations, _actions,
            # _rewards, _is_done, _actor_states, _critic_states
            if hasattr(v, '__len__'):
                data_group.create_dataset(k, data=v, compression="lzf")  # can compress only array-like structures
            else:
                data_group.create_dataset(k, data=v)  # can't compress scalars
        f.close()
        return

    def save_exp_replay(self, epoch):
        print('saving exp_replay...')
        self.save_experience_replay_as_h5(self.experience_replay, self.logdir + 'exp_replay_{}.h5'.format(epoch))
        # with open(self.logdir + 'exp_replay_{}.pickle'.format(epoch), 'wb') as f:
        #     pickle.dump(self.experience_replay, f)

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

    def _test_agent(self, render): 
        # tests only non-shaped reward
        episode_reward = 0.0
        observation, done = self.test_env.reset(), False
        agent_state = None
        if render:
            self.test_env.render()
        while not done:
            action, agent_state = self.agent.act_test(observation, agent_state)
            observation, reward, done, _ = self.test_env.step(action)
            if render:
                self.test_env.render()
            episode_reward += reward 
        return episode_reward


    def test_n(self, n, render):
        print('testing agent...')
        self.agent.actor_eval()
        mean_total_reward = 0.0
        
        for i in range(n):
            episode_reward = self._test_agent(render)
            if render:
                print('episode {} reward: {}'.format(i, episode_reward))
            mean_total_reward += episode_reward
           
        mean_total_reward /= n
    
    
        return mean_total_reward



    def sample_new_experience(self):

        # new_segment.segment - obs: (n_env, T+1, dim), act, rew: (n_env, T, dim), don: (n_env, T). T==10.
        # priority: (n_env,)        
        new_segment, priority = self.segment_sampler.sample()

        self.experience_replay.push(
            new_segment, priority, self.priority_exponent
        )

        
        self.collected_experience += 1




    def _train_step(self, batch_size,
                     learn_policy # True.
                     ):

        # sample a batch of segments of data.
        segment_data, exp_ids, importance_weights = self.experience_replay.sample(batch_size, self.importance_exponent)
       
        segment, actor_state, critic_state = segment_data
        
        losses, q_min, upd_priority = self.agent.learn_from_data( # (5,), (q_dim,), (b,)
            segment, importance_weights, actor_state, critic_state, learn_policy
        )

        self.experience_replay.update_priorities(
            exp_ids, upd_priority, self.priority_exponent
        )

        return losses, q_min # (5,), (q_dim,)


 
    def _train_epoch(self,
                     epoch, 
                     epoch_size, # 500.
                     train_steps, # 16.
                     batch_size, # 256.
                     importance_delta, priority_delta,
                     learn_policy=True # True.
                     ):
        
        self.agent.train()

        for _ in trange(epoch_size, desc='epoch_{}'.format(epoch)): 
 
            self.sample_new_experience() # will push to replay buffer.

            for train_step in range(train_steps):
                self._train_step(batch_size, learn_policy) # (5,), (q_dim,)
        
            self.importance_exponent += importance_delta
            self.importance_exponent = min(self.end_importance_exponent, self.importance_exponent)

            self.priority_exponent += priority_delta
            self.priority_exponent = min(self.end_priority_exponent, self.priority_exponent)



    # batch size here is the number of segments to sample from experience replay
    def train(self,
              min_experience_len, # 100.
              num_epochs, # 40.
              epoch_size,
              train_steps, batch_size,
              test_n, render,
              prioritization_steps, # 3000.
              pretrain_critic=False,
              start_exp_replay_file=None):

        # end_priority_exponent: 0.9.
        # priority_exponent: 0.2.
        priority_delta = (self.end_priority_exponent - self.priority_exponent) / prioritization_steps
        importance_delta = (self.end_importance_exponent - self.importance_exponent) / prioritization_steps

        self.agent.train() # not train, but set to train mode.

        self.segment_sampler.sample_first_half_segment()


        if start_exp_replay_file is not None: # no.
            self.load_exp_replay(start_exp_replay_file)
        else: # yes.
            print('filling buffer...')
            for _ in trange(min_experience_len): # 100.
                self.sample_new_experience() # will push to replay buffer.
            self.save_exp_replay(0)



        if pretrain_critic: # no
            print('pretraining...')
            self._train_epoch(
                -1, epoch_size // 2, train_steps, batch_size,
                0, 0, False
            )



        print('training...')
        for epoch in range(num_epochs):

            self._train_epoch(
                epoch, epoch_size, train_steps, batch_size,
                importance_delta, priority_delta
            )

            # if self.collected_experience % self.exp_replay_save_frequency == 0:
            self.save_exp_replay(epoch + 1)  # save exp_replay and agent every epoch
            self.agent.save(self.logdir + 'epoch_{}.pth'.format(epoch))

            # test after saving
            test_reward = self.test_n(test_n, render)

            if test_reward > self.best_reward:
                self.best_reward = test_reward
                self.agent.save(self.logdir + 'best_test_reward.pth')
 


    def pretrain_from_segments(self, segments_file, n_epoch, batch_size, actor_size=1, critic_size=1):
        print('pretraining for {} epochs'.format(n_epoch))
        with open(segments_file, 'rb') as f:
            segments = pickle.load(f)
        num_segments = len(segments)
        for epoch in trange(n_epoch, desc='pretraining'):
            np.random.shuffle(segments)
            for i in range(0, num_segments, batch_size):
                data_batch = segments[i:i + batch_size]
                current_bs = len(data_batch)
                data_batch = list(map(np.array, zip(*data_batch)))
                actor_state = np.zeros((current_bs, actor_size), dtype=float)
                critic_state = np.zeros((current_bs, critic_size), dtype=float)
                losses, _, priority = self.agent.learn_from_data(
                    data_batch, 1.0, actor_state, critic_state
                )
                # add data from the segment file to the experience replay
                if epoch == n_epoch - 1:
                    self.experience_replay.push(
                        (data_batch, actor_state, critic_state),
                        priority, self.priority_exponent
                    )
               
        self.agent.save(self.logdir + 'pretraining.pth')
        print('pretraining done')

    def load_checkpoint(self, agent_checkpoint_file, load_full):
        if agent_checkpoint_file is not None:
            if load_full:
                self.agent.load(agent_checkpoint_file)
            else:
                self.agent.load_policy(agent_checkpoint_file)

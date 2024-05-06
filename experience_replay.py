import pickle
import numpy as np
from time import time

from parameters import observation_shape, action_shape
from parameters import train_env_args
from parameters import critic_args


class SegmentTree:
    def __init__(
            self, 
            capacity, # 0.25e6. 
            reward_dim, # 6.       
    ):
        self._capacity = capacity
        self._index = 0
        self._full = False


        # _sum_tree: idx -> priority.
        self._sum_tree = np.zeros((2 * capacity - 1,), dtype=np.float32)


        segment_len = train_env_args.segment_len

        # 250000 * (11 * 339 + 10*(22+6+1)) * 4 Bytes
        self._observations = np.zeros((capacity, segment_len + 1, sum(observation_shape)), dtype=np.float32)
        self._actions = np.zeros((capacity, segment_len, action_shape), dtype=np.float32)
        self._rewards = np.zeros((capacity, segment_len, reward_dim), dtype=np.float32)
        self._is_done = np.zeros((capacity, segment_len), dtype=np.float32)
 
  

    def _propagate(self, index):

        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self._sum_tree[parent] = self._sum_tree[left] + self._sum_tree[right]

        if parent != 0:
            self._propagate(parent)


    def update(self, index, priority):
        self._sum_tree[index] = priority
        self._propagate(index)


    def _append(self, data):

        # obs: (T+1, dim), action, reward: (T, dim), done: (T). T==10.
        obs, action, reward, done = data

        self._observations[self._index] = obs
        self._actions[self._index] = action
        self._rewards[self._index] = reward
        self._is_done[self._index] = done
 
    def append(self, data, priority):
        '''
        priority: (,).
        '''
        self._append(data)
    
        self.update(self._index + self._capacity - 1, priority)
        self._index = (self._index + 1) % self._capacity
        self._full = self._full or self._index == 0


    def _retrieve(self, index, value): # find a leaf node with priority => `value` starting from `index`.

        left, right = 2 * index + 1, 2 * index + 2

        if left >= len(self._sum_tree):
            return index
        
        elif value <= self._sum_tree[left]:
            return self._retrieve(left, value)
        
        else:
            return self._retrieve(right, value - self._sum_tree[left])


    def _get_data_by_idx(self, idx):
      
        data = (
            self._observations[idx],
            self._actions[idx],
            self._rewards[idx],
            self._is_done[idx]        
        )        
        return data


    def find(self, value):

        # find a leaf node with priority => `value` starting from `index`.
        index = self._retrieve(0, value)

        data_index = index - self._capacity + 1

        # data, current priority, data_index, tree index.
        result = (
            # self.tree[data_index % self._capacity],
            self._get_data_by_idx(data_index % self._capacity),
            self._sum_tree[index], # current priority.
            data_index, 
            index
        )
        return result


    def get(self, data_index):
        data = self._get_data_by_idx(data_index % self._capacity)
        return data


    def total(self): # sum of priorities of all data.
        return self._sum_tree[0]


    def save_raw_data(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(
                {
                    'observations': self._observations,
                    'actions': self._actions,
                    'rewards': self._rewards,
                    'is_done': self._is_done
                }, f
            )




class PrioritizedExperienceReplay:

    def __init__(self, capacity=250000):
 

        self.capacity = capacity
        self.tree = SegmentTree(capacity, critic_args.q_value_dim)


    def push(self, segment, priority_loss, alpha):
 
             
        # obs: (n_env, T+1, dim), action, reward: (n_env, T, dim), done: (n_env, T). T==10.
        obs, action, reward, done = segment

        for i in range(len(priority_loss)): # priority_loss: (n_env,) 

            # if not done[i][0]:  # prevent appending 'broken' segments
            self.tree.append(
                (obs[i], action[i], reward[i], done[i]),  # segment itself                
                priority_loss[i] ** alpha
            )
 

    def _get_sample_from_segment(self, interval_size, i):
        '''
        find one data with priority in the range [i * interval_size, (i + 1) * interval_size].
        '''

        valid = False

        while not valid:

            sample = np.random.uniform(i * interval_size, (i + 1) * interval_size) # float.

            # data, current priority, data_index, tree index.
            data, priority, idx, tree_idx = self.tree.find(sample)

            if priority != 0:
                valid = True

        return data, priority, idx, tree_idx



    def sample(self, batch_size, beta):

   
        p_total = self.tree.total() # sum of priorities of all data.
        interval_size = p_total / batch_size


        zip_data = [self._get_sample_from_segment(interval_size, i) for i in range(batch_size)]

        data, priority, ids, tree_ids = zip(*zip_data)

   
        
        segment = map(np.array, zip(*data))
 
        probs = np.array(priority) / p_total

        importance_weights = (self.capacity * probs) ** beta
        importance_weights /= importance_weights.max()
 
 
        return segment, tree_ids, importance_weights




    def update_priorities(self, ids, priority_loss, alpha):
  
        priorities = np.power(priority_loss, alpha)
        [self.tree.update(idx, priority) for idx, priority in zip(ids, priorities)]



 
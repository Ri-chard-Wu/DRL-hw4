
import numpy as np

 

class Memory:

    def __init__(self, max_size, observation_shape, action_shape, dtype='float32'):
        self.observation_shape = observation_shape
        
        self.action_shape = action_shape
        self.dtype = dtype

        self.obs      = np.zeros((max_size, *self.observation_shape)).astype(dtype)
        self.actions  = np.zeros((max_size, *self.action_shape)).astype(dtype)
        self.rewards  = np.zeros((max_size, 1)).astype(dtype)
        self.dones    = np.zeros((max_size, 1)).astype(dtype)
        self.next_obs = np.zeros((max_size, *self.observation_shape)).astype(dtype)

        self.max_size = max_size
        self.pointer = 0
        self.size = 0 


    def store_transition(self, obs, actions, rewards, dones, next_obs):

     

        # assert 2d arrays coming in
        obs, actions, rewards, dones, next_obs = np.atleast_2d(obs, actions, rewards, dones, next_obs)

        B = obs.shape[0]  # batched number of environments
        idxs = np.arange(self.pointer, self.pointer + B) % self.max_size
        self.obs[idxs] = obs
        self.actions[idxs] = actions
        self.rewards[idxs] = rewards
        self.dones[idxs] = dones
        self.next_obs[idxs] = next_obs

        # step buffer pointer and size
        self.pointer = (self.pointer + B) % self.max_size
        self.size = min(self.size + B, self.max_size)


    def sample(self, batch_size): 
        idxs = np.random.randint(0, self.size, batch_size) 
        return np.atleast_2d(self.obs[idxs], self.actions[idxs], self.rewards[idxs], self.dones[idxs], self.next_obs[idxs])




    def prefill(self, env, n_prefill_steps=10):
      
        obs = env.reset()

        for _ in range(n_prefill_steps):
            actions = np.random.uniform(-1, 1, (env.num_envs,) + self.action_shape) 
            next_obs, r, done, info = env.step(actions)
            r_aug = np.vstack([i.get('rewards', 0) for i in info]) # (n_env, 1)
            self.store_transition(obs, actions, r + r_aug, done, next_obs)
            obs = next_obs

            print(f'prefill: {_}')



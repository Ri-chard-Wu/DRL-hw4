import numpy as np 

 
import tensorflow as tf


class SegmentSampler:
    def __init__(
            self,
            agent, environment, segment_len,
            q_weights # [2.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    ):
        self.agent = agent
        self.environment = environment # L2M env + some wrappers.
        self.segment_len = segment_len # 10.
        self.q_weights = np.array([q_weights], dtype=float)
        

        

        observation = self.environment.reset()
        batch_size = observation.shape[0]
        self.reward = np.zeros(batch_size, dtype=float)
        self.episode_length = np.zeros(batch_size, dtype=float)

        self.previous_half_segment = None
        self.current_observation = observation
    



    def sample_first_half_segment(self):

        segment, observation = self._sample_half_segment(self.current_observation)

       
        self.previous_half_segment = segment

        self.current_observation = observation
  


    def _sample_half_segment(self, observation):
 
        observations = []
        actions = []
        rewards = []
        is_done = []

        for step in range(self.segment_len // 2): # 10 // 2 == 5.

            observations.append(observation)
 
            action, reward, observation, done = self._step(observation)

          
            # self.q_weights: np.array([2.0, 1.0, 1.0, 1.0, 1.0, 1.0]).
            # reward: (n_env, 6).
            self.reward += (self.q_weights * reward).sum(axis=-1)  # increase reward even if episode is done
            self.episode_length += (1.0 - done)  # increase episode len only for alive environments
            actions.append(action)
            rewards.append(reward)
            is_done.append(done)

            if np.any(done):
 
         
                self.reward *= (1.0 - done)
                self.episode_length *= (1.0 - done)

  

        segment = (np.array(observations), np.array(actions), np.array(rewards), np.array(is_done))
      
        return segment, observation 



    def _step(self, observation):

      
        action = self.agent.act_q(observation)
 
        new_observation, reward, done, _ = self.environment.step(action)
      
        return action, reward, new_observation, done



    def _concatenate_segments(self, segment):
        # (observation, action, reward) - vectors, (done) - scalar
        shapes = [(1, 0, 2), (1, 0, 2), (1, 0, 2), (1, 0)]

        segment = [
            np.concatenate((a, b)).transpose(c) # observation, action, reward: (n_env, T, dim), done: (n_env, T).
            for a, b, c in zip(self.previous_half_segment, segment, shapes)
        ]
        return segment

 


    def sample(self):
 

        half_segment, new_observation = self._sample_half_segment(self.current_observation)
 
  
        # obs, act, rew: (n_env, T, dim), don: (n_env, T). T==10.
        segment = self._concatenate_segments(half_segment) 


        
        segment[0] = np.concatenate( # (n_env, T+1, dim)
            (segment[0], new_observation[:, None, :]), 1
        )

        # segment - obs: (n_env, T+1, dim). act, rew: (n_env, T, dim). don: (n_env, T). T==10.
        priority_loss = self.agent.calculate_priority_loss(segment) # (n_env,).
 
     
        self.previous_half_segment = half_segment
        self.current_observation = new_observation


        # segment - obs: (n_env, T+1, dim), act, rew: (n_env, T, dim), don: (n_env, T). T==10.
        # priority_loss: (n_env,)
        return segment, priority_loss  

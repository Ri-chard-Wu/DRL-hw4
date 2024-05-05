import numpy as np 








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
        self.actor_state = None
        self.critic_state = None



    def sample_first_half_segment(self):

        first_half_segment = self._sample_half_segment(
            self.current_observation, self.actor_state, self.critic_state
        )

        segment, obs_and_state = first_half_segment
        observation, actor_state, critic_state = obs_and_state

        self.previous_half_segment = segment

        self.current_observation = observation
        self.actor_state = actor_state
        self.critic_state = critic_state




    def _sample_half_segment(self, observation, actor_state, critic_state):

         
        observations = []
        actions = []
        rewards = []
        is_done = []

        for step in range(self.segment_len // 2): # 10 // 2 == 5.

            observations.append(observation)
 
            actor_state, critic_state, action, reward, observation, done = \
                                    self._step(observation, actor_state, critic_state)

          
            # self.q_weights: np.array([2.0, 1.0, 1.0, 1.0, 1.0, 1.0]).
            # reward: (n_env, 6).
            self.reward += (self.q_weights * reward).sum(axis=-1)  # increase reward even if episode is done
            self.episode_length += (1.0 - done)  # increase episode len only for alive environments
            actions.append(action)
            rewards.append(reward)
            is_done.append(done)

            if np.any(done):


                actor_state, critic_state = self.agent.zero_state(
                    actor_state, critic_state, done
                )
                
                
                self.reward *= (1.0 - done)
                self.episode_length *= (1.0 - done)

  

        segment = (np.array(observations), np.array(actions), np.array(rewards), np.array(is_done))
        obs_and_state = (observation, actor_state, critic_state)
    
        return segment, obs_and_state 





    def _step(self, observation, actor_state, critic_state):

      
        action, new_actor_state, new_critic_state = self.agent.act_q(
            observation, actor_state, critic_state
        )
 
        new_observation, reward, done, _ = self.environment.step(action)
      
        return new_actor_state, new_critic_state, action, reward, new_observation, done



    def _concatenate_segments(self, segment):
        # (observation, action, reward) - vectors, (done) - scalar
        shapes = [(1, 0, 2), (1, 0, 2), (1, 0, 2), (1, 0)]

        segment = [
            np.concatenate((a, b)).transpose(c) # observation, action, reward: (n_env, T, dim), done: (n_env, T).
            for a, b, c in zip(self.previous_half_segment, segment, shapes)
        ]
        return segment



    def reflect_segment(self, segment):
        observation, action, reward, done = segment
        v_tgt, observation = observation[:, :, 11 * 11 * 2], observation[:, :, 11 * 11 * 2:]
        reflected_v_tgt = self.reflect_v_tgt(v_tgt)
        reflected_obs = self.reflect_observation(observation)
        reflected_obs = np.concatenate([reflected_v_tgt, reflected_obs], axis=-1)

        reflected_action = self.reflect_action(action)
        reflected_reward = np.copy(reward)
        reflected_done = np.copy(done)
        reflected_segment = (
            reflected_obs, reflected_action, reflected_reward, reflected_done
        )
        return reflected_segment


    @staticmethod
    def reflect_v_tgt(v_tgt):
  
        return v_tgt

    @staticmethod
    def reflect_observation(observation):
        # TODO: implement ASAP
        pelvis = observation[:, :, :7]
        legs = observation[:, :, 7:]
        return observation

    @staticmethod
    def reflect_action(action):
        leg_1, leg_2 = action[:, :, :11], action[:, :, 11:]
        reflected_action = np.concatenate([leg_2, leg_1], axis=-1)
        return reflected_action



    def sample(self):

        # returns (
        #          the segment,
        #          actor's & critic's states at the beginning of the segment,
        #          priority of the segment)

        half_segment, obs_and_state = self._sample_half_segment(
            self.current_observation, self.actor_state, self.critic_state)
 
 
        new_observation, actor_state, critic_state = obs_and_state

        # obs, act, rew: (n_env, T, dim), don: (n_env, T). T==10.
        segment = self._concatenate_segments(half_segment) 


        
        segment[0] = np.concatenate( # (n_env, T+1, dim)
            (segment[0], new_observation[:, None, :]), 1
        )

        # segment - obs: (n_env, T+1, dim). act, rew: (n_env, T, dim). don: (n_env, T). T==10.
        priority_loss = self.agent.calculate_priority_loss(segment, self.actor_state, self.critic_state) # (n_env,).
 
        result = (
            segment, # obs: (n_env, T+1, dim), act, rew: (n_env, T, dim), don: (n_env, T). T==10.
            self.actor_state,
            self.critic_state,
        )
 
        self.previous_half_segment = half_segment
        self.current_observation = new_observation
        self.actor_state = actor_state
        self.critic_state = critic_state

        # result.segment - obs: (n_env, T+1, dim), act, rew: (n_env, T, dim), don: (n_env, T). T==10.
        # priority_loss: (n_env,)
        return result, priority_loss  

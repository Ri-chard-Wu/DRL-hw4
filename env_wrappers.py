import time
import gym
import numpy as np
from osim.env import L2M2019Env
from multiprocessing import Process, Pipe
 
from parameters import train_env_args, test_env_args
 

 
 
 

def worker(i, remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
   
    while True:
        cmd, data = remote.recv()
        if cmd == 'step': 
            ob, reward, done, info = env.step(data)
            if info['reset']:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()     
            remote.send(ob)  
        elif cmd == 'close':
            remote.close()
            break       
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))                        
        else:
            raise NotImplementedError

 
class CloudpickleWrapper():
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class VecEnv():
 

    def __init__(self, env_fns):
        
        self.closed = False 
        self.num_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.num_envs)])
  
        self.ps = [Process(target=worker, args=(i, work_remote, remote, CloudpickleWrapper(env_fn))) for i, (work_remote, remote, env_fn) in enumerate(zip(self.work_remotes, self.remotes, env_fns))]
   
        for p in self.ps:
            p.daemon = True    
            p.start()  
        for remote in self.work_remotes:
            remote.close()
 
        self.remotes[0].send(('get_spaces', None)) 
        self.observation_space, self.action_space = self.remotes[0].recv() 
 

    def step(self, actions): # actions: (n_env, n_actions) numpy array.
        for i in range(len(self.remotes)): 
            self.remotes[i].send(('step', actions[i]))        
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results) 
        return np.stack(obs), np.stack(rews), np.stack(dones), infos


    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = np.stack([remote.recv() for remote in self.remotes])
        return obs
 
    def close(self):
        if self.closed:
            return
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


class SkeletonWrapper(gym.Wrapper):


    def __init__(self, env, is_train_env, args):

        gym.Wrapper.__init__(self, env)
        self.is_train_env = is_train_env

        self.frame_skip = args.frame_skip

        env.spec.timestep_limit = args.timestep_limit
        self.timestep_limit = args.timestep_limit
        self.episode_len = 0

        # reward weights from the environment
    

        self.alive_bonus = args.alive_bonus
        self.death_penalty = args.death_penalty
        self.task_bonus = args.task_bonus

        # placeholders
        self.step_time = 0
        self.v_tgt_step_penalty = 0

 
 
    def leg_to_numpy(self, leg):
        observation = []
        for k, v in leg.items():
            if type(v) is dict:
                observation += list(v.values())
            else:
                observation += v
            
        return np.array(observation)
 

    def observation_to_numpy(self, obs):
        v_tgt_field = obs['v_tgt_field'].reshape(-1) / 10 # (242,)

        p = obs['pelvis']
        pelvis = np.array([p['height'], p['pitch'], p['roll']] + p['vel']) # (9,)

        r_leg = self.leg_to_numpy(obs['r_leg'])  # (44,)
        l_leg = self.leg_to_numpy(obs['l_leg'])  # (44,)

        flatten_observation = np.concatenate([v_tgt_field, pelvis, r_leg, l_leg]) # (339,)
        return flatten_observation # (339,)




    def reset(self, **kwargs):
        obs = self.env.reset()

        obs = self.observation_to_numpy(obs) # (339,)
 
        assert self.env.env.env.env.d_reward['weight']['footstep'] == 10
        assert self.env.env.env.env.d_reward['weight']['effort'] == 1
        assert self.env.env.env.env.d_reward['weight']['v_tgt'] == 1
        
        self.episode_len = 0

        return obs # (339,)



    def step(self, action):

        raw_obs, reward, done, info = self.env.step(action)

        obs = self.observation_to_numpy(raw_obs) # (339,)

        if self.is_train_env: # yes.
            reward = self.shape_reward(action, reward, done)

        self.episode_len += 1
 

        return obs, reward, done, info



    @staticmethod
    def crossing_legs_penalty(state_desc):
        # stolen from Scitator
        pelvis_xyz = np.array(state_desc['body_pos']['pelvis'])
        left = np.array(state_desc['body_pos']['toes_l']) - pelvis_xyz
        right = np.array(state_desc['body_pos']['toes_r']) - pelvis_xyz
        axis = np.array(state_desc['body_pos']['head']) - pelvis_xyz
        cross_legs_penalty = np.cross(left, right).dot(axis)

        if cross_legs_penalty > 0:
            cross_legs_penalty = 0.0

        return 10 * cross_legs_penalty


    @staticmethod
    def bending_knees_bonus(state_desc):
        # stolen from Scitator
        r_knee_flexion = np.minimum(state_desc['joint_pos']['knee_r'][0], 0.)
        l_knee_flexion = np.minimum(state_desc['joint_pos']['knee_l'][0], 0.)
        # bonus only for one bended knee
        # bend_knees_bonus = np.abs(r_knee_flexion + l_knee_flexion)
        bend_knees_bonus = -np.minimum(r_knee_flexion, l_knee_flexion)
        # I believe 0.4 is optimal clip value
        bend_knees_bonus = np.clip(bend_knees_bonus, 0.0, 0.4)
        return bend_knees_bonus


    @staticmethod
    def get_v_body(state_desc):
        dx = state_desc['body_vel']['pelvis'][0]
        dy = state_desc['body_vel']['pelvis'][2]
        return np.asarray([dx, -dy])



    def get_v_tgt(self, state_desc):
        p_body = [state_desc['body_pos']['pelvis'][0], -state_desc['body_pos']['pelvis'][2]]
        v_tgt = self.env.env.env.env.vtgt.get_vtgt(p_body).T
        return v_tgt



    @staticmethod
    def pelvis_velocity_bonus(v_tgt, v_body):
        v_tgt_abs = (v_tgt ** 2).sum() ** 0.5
        v_body_abs = (v_body ** 2).sum() ** 0.5
        v_dp = np.dot(v_tgt, v_body)
        cos = v_dp / (v_tgt_abs * v_body_abs + 0.1)
        bonus = v_body_abs / 1.4
        return (np.sign(cos) * bonus)[0]



    @staticmethod
    def target_achieve_bonus(v_tgt):
        v_tgt_square = (v_tgt ** 2).sum()
        if 0.5 ** 2 < v_tgt_square <= 0.7 ** 2:
            return 0.1
        elif v_tgt_square <= 0.5 ** 2:
            return 1.0 - 3.5 * v_tgt_square
        else:
            return 0
 

    @staticmethod
    def dense_effort_penalty(action):
        action = 0.5 * (action + 1)  # transform action from nn to environment range
        effort = 0.5 * (action ** 2).mean()
        return -effort



    def v_tgt_deviation_penalty(self, foot_step_reward, v_body, v_tgt):
        delta_v = v_body - v_tgt
        penalty = 0.5 * np.sqrt((delta_v ** 2).sum())
        self.v_tgt_step_penalty += penalty
        if foot_step_reward != 0.0:
            step_penalty = -self.v_tgt_step_penalty / max(1, self.step_time)
            self.v_tgt_step_penalty = 0
            self.step_time = 0
            return step_penalty
        else:
            self.step_time += 1
            return 0



    def shape_reward(self, action, reward, done):

        state_desc = self.env.env.env.env.get_state_desc()

        if not self.alive_bonus: # yes.
            reward -= 0.1 * self.frame_skip

        if not self.task_bonus: # yes
            if reward >= 450:
                reward -= 500


        dead = (self.episode_len + 1) * self.frame_skip < self.timestep_limit

        if done and dead:
            reward = self.death_penalty # -50.

        v_body = self.get_v_body(state_desc)
        v_tgt = self.get_v_tgt(state_desc)
        clp = self.crossing_legs_penalty(state_desc)
        
        pvb = self.pelvis_velocity_bonus(v_tgt, v_body)
        tab = self.target_achieve_bonus(v_tgt)
        vdp = self.v_tgt_deviation_penalty(reward, v_body, v_tgt)
        dep = self.dense_effort_penalty(action)

        return [reward, clp, vdp, pvb, dep, tab]



    def get_body_pos_vel(self):
        state_desc = self.env.env.env.env.get_state_desc()
        p_body = [state_desc['body_pos']['pelvis'][0], -state_desc['body_pos']['pelvis'][2]]
        v_body = [state_desc['body_vel']['pelvis'][0], -state_desc['body_vel']['pelvis'][2]]
        v_tgt = self.env.env.env.env.vtgt.get_vtgt(p_body).T
        return p_body, v_body, v_tgt

 

class NormalizedActions(gym.ActionWrapper): # map from tanh's [-1, 1] to action_space's [low, high].
    def action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high

        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)

        return action

    def reverse_action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high

        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)

        return action


class FrameSkipWrapper(gym.Wrapper):
    def __init__(self, env, frame_skip):
        gym.Wrapper.__init__(self, env)
        self.frame_skip = frame_skip

    def reset(self):
        return self.env.reset()

    def step(self, action):
        step_reward = 0.0
        for _ in range(self.frame_skip):
            time_before_step = time.time()
            obs, reward, done, info = self.env.step(action)

            # early stopping of episode if step takes too long
            if (time.time() - time_before_step) > 10: done = True
            step_reward += reward
            if done: break
        return obs, step_reward, done, info


 

class SegmentPadWrapper(gym.Wrapper):
    def __init__(self, env, segment_len):
        gym.Wrapper.__init__(self, env)

        self.segment_len = segment_len // 2 # 5.
        self.episode_len = 0
        self.residual_len = 0

        self.episode_ended = False
        self.last_transaction = None

    def reset(self):
        self.episode_ended = False
        self.episode_len = 0
        self.residual_len = 0
        return self.env.reset()

    def check_reset(self):
        
        if (self.episode_len + self.residual_len) % self.segment_len == 0:
            self.last_transaction[3]['reset'] = True

        self.residual_len += 1


    def step(self, action):
        if self.episode_ended:
            self.check_reset()
        else:

            self.episode_len += 1

            # obs, reward, done, info
            self.last_transaction = self.env.step(action)
            self.last_transaction[3]['reset'] = False # info['reset'] = False.

            if self.last_transaction[2]: # done.
                self.episode_ended = True
                self.check_reset()

        return self.last_transaction


 



def make_env(args, seed, is_train_env=True):

    def _make_env():

        env = L2M2019Env(
            visualize=False,
            integrator_accuracy=args.accuracy,
            difficulty=args.difficulty,
            seed=seed
        )

        env = FrameSkipWrapper(env, args.frame_skip)
        if(is_train_env): env = SegmentPadWrapper(env, args.segment_len)
        env = NormalizedActions(env) # map from tanh's [-1, 1] to action_space's [low, high].
        env = SkeletonWrapper(env, is_train_env, args)

        return env

    return _make_env




def make_train_env(args=train_env_args):

    seeds = np.random.choice(1000, size=args.env_num + 1, replace=False)
 
    env = VecEnv([make_env(args, seeds[i], True) for i in range(args.env_num)])

    return env

 

def make_test_env(args=test_env_args):

    seed = np.random.choice(1000, size=1, replace=False)[0]
 
    env = make_env(args, seed, False)

    return env


















 
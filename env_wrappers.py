

import contextlib
import multiprocessing as mp

from multiprocessing import Process, Pipe



import os  

import numpy as np 

import gym
from collections import deque
 
from osim.env import L2M2019Env



class L2M2019EnvBaseWrapper(L2M2019Env):
    """ Wrapper to move certain class variable to instance variables """
    def __init__(self, **kwargs):

        self._model = kwargs.pop('model', '3D')
        stepsize = kwargs.pop('stepsize', 0.01)
        self._visualize = kwargs.get('visualize', False)

        self._osim_model = None
        super().__init__(visualize=kwargs['visualize'],
                         integrator_accuracy=kwargs['integrator_accuracy'],
                         difficulty=kwargs['difficulty'])  # NOTE -- L2M2019Env at init calls get_model_key() which pulls self.model; -> setting _model here initializes the model to 2D or 3D
        self.osim_model.stepsize = stepsize

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def change_model(self, model):
        # overwrite method so as to remove arguments `difficulty` and `seed` in the parent change_model method
        if self.model != model:
            self.model = model
            self.load_model(self.model_paths[self.get_model_key()])

    @property
    def osim_model(self):
        return self._osim_model

    @osim_model.setter
    def osim_model(self, model):
        self._osim_model = model

    @property
    def visualize(self):
        return self._visualize

    @visualize.setter
    def visualize(self, new_state):
        assert isinstance(new_state, bool)
        self._visualize = new_state

    # match evaluation env
    def step(self, action):
        return super().step(action, project=True, obs_as_dict=True)

    def reset(self, **kwargs):
        obs_as_dict = kwargs.pop('obs_as_dict', True)
        return super().reset(obs_as_dict=obs_as_dict, **kwargs)



 
class RandomPoseInitEnv(gym.Wrapper):

    def __init__(self, env=None, anneal_start_step=1000, anneal_end_step=2000, **kwargs):        
        super().__init__(env)
        # anneal pose to zero-pose
        self.anneal_start_step = anneal_start_step
        self.anneal_end_step = anneal_end_step
        self.anneal_step = 0

    def reset(self, **kwargs): # kwargs: no kwargs.
        '''
        Default init pose:

        INIT_POSE = np.array([
        
            0, # forward speed
            0, # rightward speed
            0.94, # pelvis height
            0*np.pi/180, # trunk lean

            0*np.pi/180, # [right] hip adduct
            0*np.pi/180, # hip flex
            0*np.pi/180, # knee extend
            0*np.pi/180, # ankle flex

            0*np.pi/180, # [left] hip adduct
            0*np.pi/180, # hip flex
            0*np.pi/180, # knee extend
            0*np.pi/180]) # ankle flex

        '''

        seed = kwargs.get('seed', None)
        if seed is not None: # no
            state = np.random.get_state()
            np.random.seed(seed)

        x_vel = np.clip(np.abs(np.random.normal(0, 1.5)), a_min=None, a_max=3.5) # forward speed
        y_vel = np.random.uniform(-0.15, 0.15) # rightward speed

        # foot in the air
        leg1 = [np.random.uniform(0, 0.1), np.random.uniform(-1, 0.3), np.random.uniform(-1.3, -0.5), 
                np.random.uniform(-0.9, -0.5)]
        
        # foot on the ground
        leg2 = [np.random.uniform(0, 0.1), np.random.uniform(-0.25, 0.05), np.random.uniform(-0.5, -0.25), -0.25]

        pose = [x_vel, # forward speed
                y_vel, # rightward speed
                0.94,
                np.random.uniform(-0.15, 0.15)
                ]

        if y_vel > 0: # going to right.
            pose += leg1 + leg2 # right leg + left leg.
        else: # going to left.
            pose += leg2 + leg1 # right leg + left leg.


        pose = np.asarray(pose)

        # at the end, everything is 0 (no random) except pelvit height (0.94).
        pose *= 1 - np.clip(
            (self.anneal_step - self.anneal_start_step)/(self.anneal_end_step - self.anneal_start_step), 0, 1)
        pose[2] = 0.94


        self.anneal_step += 1

        if seed is not None: # no
            np.random.set_state(state)

        return self.env.reset(init_pose=pose, **kwargs)

  


class ActionAugEnv(gym.Wrapper):
    """ transform action from tanh policies in (-1,1) to (0,1) """
 
    def step(self, action):
        return self.env.step((action + 1)/2)


 

class PoolVTgtEnv(gym.Wrapper):
    def __init__(self, env=None, **kwargs):
        super().__init__(env)

        # v_tgt_field pooling; output size = pooled_vtgt + scale of x vel
        self.v_tgt_field_size = 4   # v_tgt_field pooled size for x and y
        self.v_tgt_field_size += 1  # distance to vtgt sink

        # adjust env reference dims
        obs_dim = env.observation_space.shape[0] - 2*11*11 + self.v_tgt_field_size
        self.observation_space = gym.spaces.Box(np.zeros(obs_dim), np.zeros(obs_dim))

    def pool_vtgt(self, obs):
        '''
        the grid in vtgt is centered at the agent's current position.
        '''

        # transpose and flip over x coord to match matplotliv quiver so easier to interpret
        vtgt = obs['v_tgt_field'].swapaxes(1,2)[:,::-1,:] # (2, 11, 11): (vx & vy, y, x).

        # downsample by 2, followed by mean pool with kernel size of 2.
        pooled_vtgt = vtgt.reshape(2,11,11)[:,::2,::2].reshape(2,3,2,3,2).mean((2,4)) # (2, 3, 3)

        # pool each coordinate
        x_vtgt = pooled_vtgt[0].mean(0)  # (3,)  # pool dx (vx?) over y coord.
        y_vtgt = np.abs(pooled_vtgt[1].mean(1))  # pool dy (vy?) over x coord and return one hot indicator of the argmin

        # y turning direction (yaw tgt) = [left, straight, right]
        y_vtgt_onehot = np.zeros_like(y_vtgt)
        y_vtgt_argsort = y_vtgt.argsort()

        # if target is behind (x_vtgt is negative and y_vtgt is [0, 1, 0] ie argmin is 1, then choose second to argmin to force turn
        y_vtgt_onehot[y_vtgt_argsort[1] if (y_vtgt[1] < 1 and y_vtgt_argsort[0] == 1) else y_vtgt_argsort[0]] = 1
        
        # distance to vtgt sink
        goal_dist = np.sqrt(x_vtgt[1]**2 + y_vtgt[1]**2)
        
        # x speed tgt = [stop, go]
        x_vtgt_onehot = (goal_dist > 0.3) # (1,)
        # print('dx {:.2f}; dy {:.2f}; dxdy {:.2f}; dx_tgt {:.2f}'.format(
        #     x_vtgt[1], y_vtgt[1], np.sqrt(x_vtgt[1]**2 + y_vtgt[1]**2), dx_tgt))
        obs['v_tgt_field'] = np.hstack([x_vtgt_onehot, y_vtgt_onehot, goal_dist]) # (5,)
        return obs

 
    def step(self, action):
        o, r, d, i = self.env.step(action)
        return self.pool_vtgt(o), r, d, i

    def reset(self, **kwargs):
        o = self.env.reset(**kwargs)
        if not o:  # submission client returns False at end
            return o
        return self.pool_vtgt(o)

    def create(self):
        return self.pool_vtgt(self.env.create())




class RewardAugEnv(gym.Wrapper):
    @staticmethod
    def compute_rewards(x_vtgt_onehot, goal_dist, height, pitch, roll, dx, dy, dz, dpitch, droll, dyaw, rf, rl, ru, lf, ll, lu):
        """ note this operates on scalars (when called by env) and vectors (when called by state predictor models) """
        # NOTE -- should be left right symmetric if using symmetric memory

        rewards = {}

        # goals -- v_tgt_field sink
        # e.g. [0. , 0.165, 0.462, 0.905, 0.999] for goal dist [2. , 1.5, 1. , 0.5, 0.1]
        rewards['vtgt_dist'] = np.clip(np.tanh(1 / np.clip(goal_dist, 0.1, None) - 0.5), 0, None)  
        rewards['vtgt_goal'] = np.where(goal_dist < 0.3, 5 * np.ones_like(goal_dist), np.zeros_like(goal_dist))

        # stability -- penalize pitch and roll
        # if in different direction ie counteracting ie diff signs, then clamped to 0, otherwise positive penalty
        rewards['pitch'] = - 1 * np.clip(pitch * dpitch, 0, float('inf')) 
        rewards['roll']  = - 1 * np.clip(roll * droll, 0, float('inf'))

        # velocity -- reward dx; penalize dy and dz
        rewards['dx'] = np.where(x_vtgt_onehot == 1, 3 * np.tanh(dx), 2 * (1 - np.tanh(5*dx)**2))
        rewards['dy'] = - 2 * np.tanh(2*dy)**2
        # rewards['dz'] = - np.tanh(dz)**2

        # footsteps -- penalize ground reaction forces outward/lateral/(+) (ie agent pushes inward and crosses legs)
        # if ll is not None:
        #     rewards['grf_ll'] = - 0.5 * np.tanh(10*ll)
        #     rewards['grf_rl'] = - 0.5 * np.tanh(10*rl)

        # falling
        rewards['height'] = np.where(height > 0.70, np.zeros_like(height), -5 * np.ones_like(height))

        return rewards




    def step(self, action):
        yaw_old = self.get_state_desc()['joint_pos']['ground_pelvis'][2]

        obs, rew, don, inf = self.env.step(action)

        # extract data
        # split to produce 3 arrays of shape (1,), (3,) and (1,)  where n is half the pooled v_tgt_field
        # v_tgt_field_size: 5.
        x_vtgt_onehot, y_vtgt_onehot, goal_dist = np.split(obs['v_tgt_field'], [1, self.v_tgt_field_size - 1], axis=-1)  
        


        # pelvis height, pelvis pitch, pelvis roll, vel (forward), vel (leftward), vel (upward), angular vel (pitch), angular vel (roll), angular vel (yaw)
        height, pitch, roll, [dx, dy, dz, dpitch, droll, dyaw] = obs['pelvis'].values()


        yaw_new = self.get_state_desc()['joint_pos']['ground_pelvis'][2]
        rf, rl, ru = obs['r_leg']['ground_reaction_forces'] # forward, rightward, upward.
        lf, ll, lu = obs['l_leg']['ground_reaction_forces'] # forward, leftward, upward.

        # convert to array for compute_rewards fn
        goal_dist = np.asarray(goal_dist)
        height = np.asarray(height)
        dx = np.asarray(dx)

        # compute rewards
        rewards = self.compute_rewards(x_vtgt_onehot, goal_dist, height, pitch, roll, dx, dy, dz, dpitch, droll, dyaw, rf, rl, ru, lf, ll, lu)

        # turning -- reward turning towards the v_tgt_field sink;
        # yaw target is relative to current yaw; so reward if the change in yaw is in the direction and magnitude of the tgt
        delta_yaw = yaw_new - yaw_old
        yaw_tgt = 0.025 * np.array([1, 0, -1]) @ y_vtgt_onehot   # yaw is (-) in the clockwise direction
        rewards['yaw_tgt'] = 2 * (1 - np.tanh(100*(delta_yaw - yaw_tgt))**2)

        inf['rewards'] = float(sum(rewards.values()))

        return obs, rew, don, inf



class SkipEnv(gym.Wrapper):
    def __init__(self, env=None, n_skips=4):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self.n_skips = n_skips

    def step(self, action):
        total_reward = 0
        aug_reward = 0
        for _ in range(self.n_skips):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if 'rewards' in info:
                aug_reward += info['rewards']
                info['rewards'] = aug_reward
            if done:
                break

        return obs, total_reward, done, info




class Obs2VecEnv(gym.Wrapper):
 
    def obs2vec(self, obs_dict):
        # Augmented environment from the L2R challenge
        res = []

        # target velocity field (in body frame)
        res += obs_dict['v_tgt_field'].flatten().tolist()

        res.append(obs_dict['pelvis']['height'])
        res.append(obs_dict['pelvis']['pitch'])
        res.append(obs_dict['pelvis']['roll'])
        res.append(obs_dict['pelvis']['vel'][0]/self.LENGTH0)
        res.append(obs_dict['pelvis']['vel'][1]/self.LENGTH0)
        res.append(obs_dict['pelvis']['vel'][2]/self.LENGTH0)
        res.append(obs_dict['pelvis']['vel'][3])
        res.append(obs_dict['pelvis']['vel'][4])
        res.append(obs_dict['pelvis']['vel'][5])

        for leg in ['r_leg', 'l_leg']:
            res += obs_dict[leg]['ground_reaction_forces']
            res.append(obs_dict[leg]['joint']['hip_abd'])
            res.append(obs_dict[leg]['joint']['hip'])
            res.append(obs_dict[leg]['joint']['knee'])
            res.append(obs_dict[leg]['joint']['ankle'])
            res.append(obs_dict[leg]['d_joint']['hip_abd'])
            res.append(obs_dict[leg]['d_joint']['hip'])
            res.append(obs_dict[leg]['d_joint']['knee'])
            res.append(obs_dict[leg]['d_joint']['ankle'])
            for MUS in ['HAB', 'HAD', 'HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH', 'GAS', 'SOL', 'TA']:
                res.append(obs_dict[leg][MUS]['f'])
                res.append(obs_dict[leg][MUS]['l'])
                res.append(obs_dict[leg][MUS]['v'])
        return res

    def step(self, action):
        o, r, d, i = self.env.step(action)
        return self.obs2vec(o), r, d, i

    def reset(self, **kwargs):
        o = self.env.reset(**kwargs)
        if not o:  # submission client returns False at the end
            return o
        return self.obs2vec(o)

    def create(self):
        return self.obs2vec(self.env.create())

 
 
 

 
# class VecEnv(ABC):
 
#     def __init__(self, num_envs, observation_space, action_space, v_tgt_field_size):
#         self.num_envs = num_envs
#         self.observation_space = observation_space
#         self.action_space = action_space
#         self.v_tgt_field_size = v_tgt_field_size

#     @abstractmethod
#     def reset(self):
 
#         pass

#     @abstractmethod
#     def step_async(self, actions):
 
#         pass

#     @abstractmethod
#     def step_wait(self):
 
#         pass

#     def close_extras(self):
#          
#         pass

#     def close(self):
#         if self.closed:
#             return
#         self.close_extras()
#         self.closed = True

#     def step(self, actions): 
#         self.step_async(actions)
#         return self.step_wait()


 





# def worker(remote, parent_remote, env_fn_wrapper):
#     parent_remote.close()
#     env = env_fn_wrapper.x()
#     try:
#         while True:
#             cmd, data = remote.recv()
#             if cmd == 'step':
#                 ob, reward, done, info = env.step(data)
#                 if done:
#                     ob = env.reset()
#                 remote.send((ob, [reward], [done], info))
#             elif cmd == 'reset':
#                 ob = env.reset()
#                 remote.send(ob)
#             elif cmd == 'close':
#                 remote.close()
#                 break
#             elif cmd == 'get_spaces_spec':
#                 remote.send((env.observation_space, env.action_space, getattr(env, 'v_tgt_field_size', 0), env.spec))
#             else:
#                 raise NotImplementedError
#     except KeyboardInterrupt:
#         print('SubprocVecEnv worker: got KeyboardInterrupt')
#     finally:
#         env.close()




@contextlib.contextmanager
def clear_mpi_env_vars():
    """
    from mpi4py import MPI will call MPI_Init by default. If the child process has MPI environment variables,
    MPI will think that the child process is an MPI process just like the parent and do bad things such as hang.
    This context manager is a hacky way to clear those environment variables temporarily such as when we are
    starting multiprocessing Processes.
    """
    removed_environment = {}
    for k, v in list(os.environ.items()):
        for prefix in ['OMPI_', 'PMI_', 'PMIX_']:
            if k.startswith(prefix):
                removed_environment[k] = v
                del os.environ[k]
    try:
        yield
    finally:
        os.environ.update(removed_environment)




# class SubprocVecEnv(VecEnv):

#     def __init__(self, env_fns, context='spawn'):
#        
#         self.waiting = False
#         self.closed = False
#         nenvs = len(env_fns)
#         ctx = mp.get_context(context)
#         self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(nenvs)])
#         self.ps = [ctx.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn))) for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        

#         for p in self.ps:
#             p.daemon = True  
#             with clear_mpi_env_vars():
#                 p.start()

#         for remote in self.work_remotes:
#             remote.close()

#         self.remotes[0].send(('get_spaces_spec', None))
#         observation_space, action_space, v_tgt_field_size, self.spec = self.remotes[0].recv()

#         VecEnv.__init__(self, len(env_fns), observation_space, action_space, v_tgt_field_size)

#     def step_async(self, actions):
#         self._assert_not_closed()
#         for remote, action in zip(self.remotes, actions):
#             remote.send(('step', action))
#         self.waiting = True

#     def step_wait(self):
#         self._assert_not_closed()
#         results = [remote.recv() for remote in self.remotes]
#         self.waiting = False
#         obs, rews, dones, infos = zip(*results)
#         return np.stack(obs), np.stack(rews), np.stack(dones), infos

#     def reset(self):
#         self._assert_not_closed()
#         for remote in self.remotes:
#             remote.send(('reset', None))
#         return np.stack([remote.recv() for remote in self.remotes])

#     def close_extras(self):
#         self.closed = True
#         if self.waiting:
#             try:
#                 results = [remote.recv() for remote in self.remotes]
#             except EOFError:  # nothing to receive / closed by garbage collection
#                 pass
#         for remote in self.remotes:
#             remote.send(('close', None))
#         for p in self.ps:
#             p.join()

#     def _assert_not_closed(self):
#         assert not self.closed, 'Trying to operate on a SubprocVecEnv after calling close()'

#     def __del__(self):
#         if not self.closed:
#             self.close()










##########################





 
def worker(i, remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
   
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, [reward], [done], info))
        elif cmd == 'reset':
            ob = env.reset()     
            remote.send(ob)  
        elif cmd == 'get_spaces_spec':
            remote.send((env.observation_space, env.action_space, getattr(env, 'v_tgt_field_size', 0), env.spec))                
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

    # def __init__(self, env_fns, context='spawn'):
        
    #     self.closed = False 

    #     ctx = mp.get_context(context)


    #     self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(len(env_fns))])
 
    #     self.ps = [ctx.Process(target=worker, args=(i, work_remote, remote, CloudpickleWrapper(env_fn))) for i, (work_remote, remote, env_fn) in enumerate(zip(self.work_remotes, self.remotes, env_fns))]
  

    #     for p in self.ps:
    #         p.daemon = True  # if the main process crashes, we should not cause things to hang
      
    #         with clear_mpi_env_vars():
 
    #             p.start() 

 
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


        self.remotes[0].send(('get_spaces_spec', None))

        self.observation_space, self.action_space, self.v_tgt_field_size, self.spec = self.remotes[0].recv() 

        # self.reset()

        # print(observation_space, action_space, v_tgt_field_size, spec)
        


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












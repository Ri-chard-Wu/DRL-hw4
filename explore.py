import numpy as np
import tensorflow as tf

from model import MLP
from env_wrappers import RewardAugEnv





class DisagreementExploration():
    '''
    https://arxiv.org/abs/1906.04161, cite: 385.
    '''
    def __init__(self, observation_shape, action_shape, lr, state_predictor_hidden_sizes, n_state_predictors, bonus_scale, **kwargs):

        # super().__init__(observation_shape, action_shape)

        self.observation_shape = observation_shape
        self.action_shape = action_shape

        # subset of obs indices to model / use in exploration modules
        self.v_tgt_field_size = observation_shape[0] - (339 - 2*11*11)  # L2M original obs dims are 339 where 2*11*11 is the orig vtgt field size
        
        # pose_idxs start with pelvis at 0 index (obs vector after the v_tgt_field)
        self.pose_idxs = np.array([*list(range(9)),                        # pelvis       (9 obs)
                                   *list(range(12,16)),                    # joints r leg (4 obs)
                                   *list(range(20+3*11+3,20+3*11+3+4))])   # joints l leg (4 obs)
        self.pose_idxs += self.v_tgt_field_size  # offset for v_tgt_field
        self.idxs = np.hstack([list(range(self.v_tgt_field_size)), self.pose_idxs])



        self.lr = lr
        self.bonus_scale = bonus_scale

        # inputs
        self.obs_ph = tf.placeholder(tf.float32, [None, *self.observation_shape], name='obs')
        self.actions_ph = tf.placeholder(tf.float32, [None, *self.action_shape], name='actions')
        self.next_obs_ph = tf.placeholder(tf.float32, [None, *self.observation_shape], name='next_obs')

        # build graph
        # 1. networks
        state_predictors = [MLP('state_predictor_{}'.format(i), 
                                  hidden_sizes=state_predictor_hidden_sizes,
                                    output_size=len(self.idxs)) for i in range(n_state_predictors)]

        # 2. state predictor outputs
        #   select obs indices to model using the state predictors
        obs = tf.gather(self.obs_ph, self.idxs, axis=1)
        next_obs = tf.gather(self.next_obs_ph, self.idxs, axis=1)

        #   whiten obs and next obs
        obs_mean, obs_var = tf.nn.moments(obs, axes=0)
        next_obs_mean, next_obs_var = tf.nn.moments(next_obs, axes=0)

        normed_obs = (obs - obs_mean) / (obs_var**0.5 + 1e-8)
        normed_next_obs = (next_obs - next_obs_mean) / (next_obs_var**0.5 + 1e-8)


        #   predict from zero-mean unit var obs; shift and scale result by obs mean and var
        normed_pred_next_obs = [model(tf.concat([normed_obs, self.actions_ph], 1)) for model in state_predictors]
        
        self.normed_pred_next_obs = tf.stack(normed_pred_next_obs, axis=1)  # (B, n_state_predictors, obs_dim)

        self.pred_next_obs = self.normed_pred_next_obs * (obs_var[None,None,:]**0.5 + 1e-8)\
                                                + obs_mean[None,None,:]  # (B, n_state_predictors, obs_dim)

        # 2. loss
        self.loss_ops = [tf.losses.mean_squared_error(pred, normed_next_obs) for pred in normed_pred_next_obs]

        # 3. training
        optimizer = tf.train.AdamOptimizer(lr, name='sp_optimizer')
        self.train_ops = [optimizer.minimize(loss, var_list=model.trainable_vars)\
                             for loss, model in zip(self.loss_ops, state_predictors)]



    def initialize(self, sess, **kwargs):
        self.sess = sess


    def get_exploration_bonus(self, obs, actions, next_obs):
        normed_pred_next_obs = self.sess.run(self.normed_pred_next_obs, {self.obs_ph: obs, self.actions_ph: actions})
        # bonus is variance among the state predictors and along the predicted state vector
        #   ie a/ incent disagreement among the state predictors (explore state they can't model well);
        #      b/ incent exploring diverse state vectors; eg left-right leg mid-stride having opposite signs is higher var than standing / legs in same position
        return self.bonus_scale * np.var(normed_pred_next_obs, axis=(1,2))[:,None]



    def select_best_action(self, obs, actions):

        # input is obs = (n_env, obs_dim); actions = (n_samples, n_env, action_dim)
        n_samples, n_env, action_dim = actions.shape

        # reshape inputs to (n_samples*n_env, *_dim)
        obs = np.tile(obs, (n_samples, 1, 1)).reshape(-1, obs.shape[-1])
        actions = actions.reshape(-1, action_dim)

        # (n_samples*n_env, n_state_predictors, obs_dim)
        pred_next_obs = self.sess.run(self.pred_next_obs, {self.obs_ph: obs, self.actions_ph: actions})  

        # compute reward, split into 9 subarrays over the last axis.
        height, pitch, roll, dx, dy, dz, dpitch, droll, dyaw = np.split(
                        pred_next_obs[:,:,self.v_tgt_field_size: self.v_tgt_field_size+9], 9, -1)

        # split to produce 3 arrays of shape (n,), (n,) and (1,)  where n is half the pooled v_tgt_field
        # split to produce 3 arrays: [0:1], [1:self.v_tgt_field_size - 1], [self.v_tgt_field_size - 1:].
        x_vtgt_onehot, _, goal_dist = np.split(pred_next_obs[:,:,:self.v_tgt_field_size], 
                                               [1, self.v_tgt_field_size - 1], axis=-1)  
        
        rewards = RewardAugEnv.compute_rewards(x_vtgt_onehot, goal_dist, height, pitch, 
                            roll, dx, dy, dz, dpitch, droll, dyaw, None, None, None, None, None, None)
        
        rewards = np.sum([v for v in rewards.values()], 0)  # (n_samples*n_env, n_state_predictors, 1)
        rewards = np.reshape(rewards, [n_samples, n_env, -1, 1])  # (n_samples, n_env, n_state_predictors, 1)
        rewards = np.sum(rewards, 2)  # sum over state predictors; out (n_samples, n_env, 1)

        actions = actions.reshape([n_samples, n_env, action_dim])

        # rewards.argmax(0)[None,...]: (1, n_env, 1)
        best_actions = np.take_along_axis(actions, rewards.argmax(0)[None,...], 0)  # out (1, n_env, action_dim)

        return np.squeeze(best_actions, 0)



    def train(self, memory, batch_size):
        losses = []
        for loss_op, train_op in zip(self.loss_ops, self.train_ops):
            batch = memory.sample(batch_size)
            loss, _ = self.sess.run([loss_op, train_op],
                        feed_dict={self.obs_ph: batch.obs, self.actions_ph: batch.actions, self.next_obs_ph: batch.next_obs})
            losses.append(loss)
        return np.mean(loss)
 
 
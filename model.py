from functools import partial

import numpy as np
import tensorflow as tf
# import tensorflow_probability as tfp

import os 
import pickle
# def mlp(x, hidden_sizes, output_size, activation, output_activation, output_kernel_initializer, output_bias_initializer):

#     for h in hidden_sizes:
#         x = tf.layers.dense(x, units=h, activation=activation)

#     return tf.layers.dense(x, units=output_size, activation=output_activation,
#             kernel_initializer=output_kernel_initializer, bias_initializer=output_bias_initializer)


# class MLP:
#     def __init__(self, name, hidden_sizes, output_size, activation=tf.nn.relu, output_activation=None, output_kernel_initializer=None, output_bias_initializer=None):

#         self.name = name

#         self.network = partial(mlp, hidden_sizes=hidden_sizes, output_size=output_size,
#                                 activation=activation, output_activation=output_activation,
#                                 output_kernel_initializer=output_kernel_initializer,
#                                 output_bias_initializer=output_bias_initializer)

#     def __call__(self, obs):
#         with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
#             x = self.network(obs)
#         return x
 
 


class MLP(tf.keras.Model):
    def __init__(self, name, hidden_sizes, output_size, activation=tf.nn.relu, output_activation=None, output_kernel_initializer=None, output_bias_initializer=None):
        
        super().__init__()

 
        self.seq = []

        for h in hidden_sizes:
            self.seq.append(tf.keras.layers.Dense(h, activation=activation))
            
        self.seq.append(tf.keras.layers.Dense(output_size, activation=output_activation, kernel_initializer=output_kernel_initializer, bias_initializer=output_bias_initializer))
 

    def call(self, x):

        for layer in self.seq:
            x = layer(x)
        
        return x


# --------------------
# BNAF implementation
# --------------------
 
class MaskedLinear(tf.keras.layers.Layer):
    def __init__(self, in_features, out_features, data_dim):
        super().__init__()
        self.data_dim = data_dim
        self.in_features = in_features
        self.out_features = out_features


    def build(self, input_shape):

        data_dim = self.data_dim
        in_features = self.in_features 
        out_features = self.out_features


        weight = np.zeros((out_features, in_features))
        mask_d = np.zeros_like(weight)
        mask_o = np.zeros_like(weight)

        for i in range(data_dim):
            # select block slices
            h     = slice(i * out_features // data_dim, (i+1) * out_features // data_dim)
            w     = slice(i * in_features // data_dim,  (i+1) * in_features // data_dim)
            w_row = slice(0,                            (i+1) * in_features // data_dim)

            # initialize block-lower-triangular weight and construct block diagonal mask_d and lower triangular mask_o
            fan_in = in_features // data_dim
            weight[h, w_row] = np.random.uniform(-np.sqrt(1/fan_in), np.sqrt(1/fan_in), weight[h, w_row].shape)
            mask_d[h, w] = 1
            mask_o[h, w_row] = 1

        mask_o = mask_o - mask_d

        self.weight = self.add_variable('weight', weight.shape, tf.float32, initializer=
                        tf.initializers.constant(weight))

        self.logg = self.add_variable('logg', [out_features, 1], tf.float32, initializer=
                        tf.initializers.constant(np.log(np.random.rand(out_features, 1))))

        self.bias = self.add_variable('bias', [out_features], tf.float32, initializer=tf.initializers.constant(
                        np.random.uniform(-1/np.sqrt(in_features), 1/np.sqrt(in_features))))

        self.mask_d = self.add_variable('mask_d', mask_d.shape, tf.float32, initializer=
                        tf.initializers.constant(mask_d), trainable=False)

        self.mask_o = self.add_variable('mask_o', mask_o.shape, tf.float32, initializer=
                        tf.initializers.constant(mask_o), trainable=False)


    def call(self, inputs):

        x, sum_logdets = inputs

        # 1. compute BNAF masked weight eq 8
        v = tf.exp(self.weight) * self.mask_d + self.weight * self.mask_o # (out_features, in_features)


        # 2. weight normalization
        v_norm = tf.norm(v, ord=2, axis=1, keepdims=True) # (out_features, 1)
        w = tf.exp(self.logg) * v / v_norm # (out_features, in_features)

        # 3. compute output and logdet of the layer
        out = tf.matmul(x, w, transpose_b=True) + self.bias
        logdet = self.logg + self.weight - 0.5 * tf.math.log(v_norm**2)
        logdet = tf.boolean_mask(logdet, tf.cast(self.mask_d, tf.uint8))
        logdet = tf.reshape(logdet, [1, self.data_dim, out.shape[1]//self.data_dim, x.shape[1]//self.data_dim])
        logdet = tf.tile(logdet, [tf.shape(x)[0], 1, 1, 1])  # output (B, data_dim, out_dim // data_dim, in_dim // data_dim)


        sum_logdets = tf.math.reduce_logsumexp(tf.transpose(sum_logdets, [0,1,3,2]) + logdet, axis=-1, keepdims=True)

        return out, sum_logdets






class Tanh(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        x, sum_logdets = inputs
        # derivation of logdet:
        # d/dx tanh = 1 / cosh^2; cosh = (1 + exp(-2x)) / (2*exp(-x))
        # log d/dx tanh = - 2 * log cosh = -2 * (x - log 2 + log(1 + exp(-2x)))
        logdet = -2 * (x - np.log(2) + tf.nn.softplus(-2.*x))
        sum_logdets = sum_logdets + tf.reshape(logdet, tf.shape(sum_logdets))
        return tf.tanh(x), sum_logdets




class FlowSequential(tf.keras.Sequential):
    def __init__(self, *args, **kwargs):
        gated = kwargs.pop('gated')
        if gated:
            self.gate = tf.Variable(tf.random_normal([1]))
        super().__init__(*args, **kwargs)

    def call(self, x):
        out = x
        sum_logdets = tf.zeros([1, tf.shape(x)[-1], 1, 1], tf.float32)

        for l in self.layers:
            out, sum_logdets = l((out, sum_logdets))

        if hasattr(self, 'gate'):
            gate = tf.sigmoid(self.gate)
            out = gate * out + (1 - gate) * x
            sum_logdets = tf.nn.softplus(sum_logdets + self.gate) - tf.nn.softplus(self.gate)

        return out, tf.squeeze(sum_logdets, [2, 3])




class BNAF(tf.keras.Model):

    '''
        BNAF (Block Neural Autoregressive Flow).
        https://arxiv.org/abs/1904.04676
    '''

    def __init__(self, hidden_sizes, output_size, n_flows=1):
        assert all(h % output_size == 0 for h in hidden_sizes), 'Size of hidden layer must divide output (actions) dim.'
        super().__init__()

        # construct model
        self.flow = []
        for i in range(n_flows):

            modules = []

            modules += [MaskedLinear(output_size, hidden_sizes[0], output_size), Tanh()]

            for h in hidden_sizes:
                modules += [MaskedLinear(h, h, output_size), Tanh()]

            modules += [MaskedLinear(h, output_size, output_size)]
            modules += (i + 1 == n_flows)*[Tanh()]  # final output only -- policy outputs actions in [-1,1]
            self.flow += [FlowSequential(modules, gated=True if i + 1 != n_flows else False)]


    def call(self, x):
        sum_logdets = 0
        for i, f in enumerate(self.flow):
            x, logdet = f(x)
            x = x[:, ::-1] if i + 1 < len(self.flow) else x   # reverse ordering between intermediate flow steps
            sum_logdets += logdet
        return x, sum_logdets



# --------------------
# Normalizing flow based policy
# --------------------

class FlowPolicy(tf.keras.Model):
    def __init__(self, name, hidden_sizes, output_size, activation=None, **kwargs):
        super().__init__()
        self.output_size = output_size # == action_dim.

        # base distribution of the flow
        # self.base_dist = tfp.distributions.Normal(loc=tf.zeros([output_size], tf.float32),
        #                                           scale=tf.ones([output_size], tf.float32))

        self.base_dist = tf.compat.v1.distributions.Normal(tf.zeros([output_size], tf.float32), tf.ones([output_size], tf.float32))

        # affine transform of state to condition the flow
        # self.affine = tf.keras.layers.Dense(2*output_size, kernel_initializer='zeros', bias_initializer='zeros')
        self.affine = MLP('base', hidden_sizes=hidden_sizes, output_size=2*output_size, **kwargs)

        # normalizing flow on top of the base distribution
        # self.flow = BNAF(hidden_sizes, output_size)
        self.flow = BNAF([4*output_size for _ in range(2)], output_size)


    def call(self, obs, n_samples=1):

        # sample actions from base distribution
        raw_actions = self.base_dist.sample([n_samples, tf.shape(obs)[0]])  # (n_samples, n_obs, action_dim)

        # affine transform conditions on state
        mu, logstd = tf.split(self.affine(obs), num_or_size_splits=2, axis=1)

        logstd = tf.clip_by_value(logstd, -20, 2) # (n_obs, action_dim)

        actions = mu + tf.exp(logstd) * raw_actions # (n_samples, n_obs, action_dim)

        actions = tf.reshape(actions, [-1, self.output_size]) # (n_samples*n_obs, action_dim)
        # actions, sum_logdet = self.tanh((actions, logstd))

        # apply flow
        actions, sum_logdet = self.flow(actions)

        # print(f'## sum_logdet.shape: {sum_logdet.shape}, logstd.shape: {logstd.shape}')
        # sum_logdet += logstd
        sum_logdet += tf.tile(logstd, [n_samples,1])

        logprob = tf.reduce_sum(self.base_dist.log_prob(actions) - sum_logdet, 1)

        # reshape to (n_samples, B, action_dim)
        actions = tf.reshape(actions, [n_samples, -1, self.output_size])
        logprob = tf.reshape(logprob, [n_samples, -1, 1])
        return actions, logprob # (n_samples, B, action_dim) and (n_samples, B, 1)
 






class V():

    def __init__(self, value_hidden_sizes, lr, tau):  
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)     
        self.tau = tau   
        
        self.v = MLP('v', hidden_sizes=value_hidden_sizes, output_size=1)
        self.v_tgt = MLP('v_tgt', hidden_sizes=value_hidden_sizes, output_size=1)

  
    def train_step(self, obs, target_v):          
        with tf.GradientTape() as tape: 
            v = self.v(obs)  
            loss = tf.math.reduce_mean(tf.norm(v - target_v, axis=1, ord=1), axis=0)
     
        grads = tape.gradient(loss, self.v.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.v.trainable_variables))
        return loss.numpy()
    
    def update_target(self): 
        for tgt, src in zip(self.v_tgt.trainable_variables, self.v.trainable_variables):
            tgt.assign((1 - self.tau) * tgt + self.tau * src)
 
    def predict(self, obs):
        return self.v_tgt(obs)




    def get_state_dict(self):

        state_dict = {'v':{}, 'v_tgt': {}}

        for i, w in enumerate(self.v.weights):                    
            state_dict['v'][w.name] = w

        for i, w in enumerate(self.v_tgt.weights):                    
            state_dict['v_tgt'][w.name] = w

        return state_dict



    def load_state_dict(self, state_dict):
 
        for i, w in enumerate(self.v.weights):                                
            w.assign(state_dict['v'][w.name])

        for i, w in enumerate(self.v.weights):                                
            w.assign(state_dict['v_tgt'][w.name])
 

class Q():
     
    def __init__(self, name, q_hidden_sizes, lr):   
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.q = MLP(name, hidden_sizes=q_hidden_sizes, output_size=1)
         
    def train_step(self, obs, act, target_q): 
        with tf.GradientTape() as tape:  
            q = self.q(tf.concat([obs, act], 1))         
            loss = tf.reduce_mean(tf.norm(q - target_q, axis=1, ord=1), axis=0) 
                 
        grads = tape.gradient(loss, self.q.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q.trainable_variables))
        
        return loss.numpy()

    def predict(self, obs, act):
        return self.q(tf.concat([obs, act], 1))


    def get_state_dict(self):

        state_dict = {'q': {}}
        for i, w in enumerate(self.q.weights):                    
            state_dict['q'][w.name] = w
    
        return state_dict


    def load_state_dict(self, state_dict):
 
        for i, w in enumerate(self.q.weights):                                
            w.assign(state_dict[w.name])
 


class DoubleQ():
     
    def __init__(self, q_hidden_sizes, lr):   
        self.q1 = Q('q1', q_hidden_sizes, lr)
        self.q2 = Q('q2', q_hidden_sizes, lr)
         
    def train_step(self, obs, act, target_q): 
        losses = 0
        losses += self.q1.train_step(obs, act, target_q)  
        losses += self.q2.train_step(obs, act, target_q) 
        return losses
        
    def predict(self, obs, act):
        q = tf.math.minimum(self.q1.predict(obs, act), self.q2.predict(obs, act))
        return q


    def get_state_dict(self):
        state_dict = {}

        state_dict['q1'] = self.q1.get_state_dict()
        state_dict['q2'] = self.q2.get_state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        self.q1.load_state_dict(state_dict['q1'])
        self.q2.load_state_dict(state_dict['q2'])





class SAC:

    # args.
    def __init__(self, observation_shape, action_shape, v_tgt_field_size, args):  

        super().__init__()

        self.args = args
 
        self.v = V(args.value_hidden_sizes, args.lr, args.tau)

        self.doubleQ = DoubleQ(args.q_hidden_sizes, args.lr)
 
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=args.policy_lr)   
        
        self.policy = FlowPolicy('policy', hidden_sizes=args.policy_hidden_sizes, output_size=action_shape[0], output_kernel_initializer='zeros', output_bias_initializer='zeros')

        self.beta = tf.Variable(tf.zeros(1), trainable=True, dtype=tf.float32, name='beta')     
        self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)      
        self.target_entropy = - 1 * np.sum(action_shape)

  
    def train_step(self, batch):   

        obs, act, rew, don, obs_next = batch
 
        obs = tf.convert_to_tensor(obs, tf.float32)
        act = tf.convert_to_tensor(act, tf.float32)
        rew = tf.convert_to_tensor(rew, tf.float32)
        don = tf.convert_to_tensor(don, tf.float32)
        obs_next = tf.convert_to_tensor(obs_next, tf.float32)



        # train policy.
        alpha = tf.math.exp(self.beta)
        with tf.GradientTape() as tape: 
            act_pred, log_pis = self.policy(obs)  # (1, batch_size, action_dim) and (...,1)
            act_pred, log_pis = tf.squeeze(act_pred, 0), tf.squeeze(log_pis, 0) 
            q_at_policy_action = self.doubleQ.predict(obs, act_pred)             
            policy_loss = tf.reduce_mean(alpha * log_pis - q_at_policy_action)
          
        grads = tape.gradient(policy_loss, self.policy.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))

        # train v.
        v_loss = self.v.train_step(obs, -policy_loss)

        # train doubleQ.
        next_v = self.v.predict(obs_next)
        target_q = rew + self.args.discount * next_v * (1 - don)
        
        q_loss = self.doubleQ.train_step(obs, act, target_q)
 

        with tf.GradientTape() as tape: 
            alpha = tf.math.exp(self.beta)
            alpha_loss = tf.reduce_mean(- alpha * log_pis - alpha * self.target_entropy)
   
        grads = tape.gradient(alpha_loss, [self.beta])
        self.alpha_optimizer.apply_gradients(zip(grads, [self.beta]))

        self.v.update_target()

        return policy_loss.numpy(), v_loss, q_loss, alpha_loss.numpy()

    def get_actions(self, obs, n_samples=1):

        obs = tf.convert_to_tensor(obs, tf.float32)

        act, _ = self.policy(obs, n_samples)

        return act.numpy()
    

    def load_state_dict(self, state_dict): 

        self.v.load_state_dict(state_dict['v'])
        self.doubleQ.load_state_dict(state_dict['doubleQ'])
        self.beta = state_dict['beta']
 
        for i, w in enumerate(self.policy.weights):                                
            w.assign(state_dict['policy'][w.name])

 

    def get_state_dict(self): 
 
        state_dict = {}
        state_dict['v'] = self.v.get_state_dict()
        state_dict['doubleQ'] = self.doubleQ.get_state_dict()        
        state_dict['beta'] = self.beta
 
        state_dict['policy'] = {}    
        for i, w in enumerate(self.policy.weights):                    
            state_dict['policy'][w.name] = w

        return state_dict

 

    def save(self, dir_name, name):
 
        if(not os.path.exists(dir_name)): 
            os.makedirs(dir_name, exist_ok=True)

        path = os.path.join(dir_name, name)

        state_dict = self.get_state_dict()
        with open(path, 'wb') as f:
            pickle.dump(state_dict, f)             

        print(f'saved ckpt: {path}')



    def load(self, dir_name, name):
        
        
        path = os.path.join(dir_name, name)
        with open(path, 'rb') as f:            
            state_dict = pickle.load(f)                    

        self.load_state_dict(state_dict)

        print(f'loaded ckpt: {path}')
 
import tensorflow as tf  

from parameters import observation_shape, action_shape
from parameters import actor_args, critic_args


 
afns = {
    'relu': tf.keras.layers.ReLU,
    'elu': tf.keras.layers.ELU
}
 
    
class Layer(tf.keras.Model):

    def __init__(self, in_features, out_features, layer_norm, afn, residual=True, drop=0.0):        
        super().__init__()
 
        seq = []

        seq.append(tf.keras.layers.Dense(out_features))
        
        if layer_norm:
            seq.append(tf.keras.layers.LayerNormalization(epsilon=1e-5))

        if afn is not None:
            seq.append(afns[afn]())
 
        if drop != 0.0:
            seq.append(tf.keras.layers.Dropout(drop))

        self.seq = tf.keras.Sequential(seq)

        self.residual = residual and in_features == out_features


    def call(self, x_in):
        
        x = self.seq(x_in)

        if self.residual:
            x = x + x_in

        return x



class PolicyNet(tf.keras.Model):

    def __init__(self, args=actor_args):
        super().__init__() 

        h = args.hidden_dim
        ln = args.layer_norm
        afn = args.afn
        res = args.residual
        drop = args.dropout
         
        tgt_dim, obs_dim = observation_shape

        self.seq = tf.keras.Sequential([
            Layer(obs_dim + tgt_dim, h, ln, afn, res, drop),
            Layer(h, h, ln, afn, res, drop),
            Layer(h, h, ln, afn, res, drop),
        ])

        self.mean = Layer(h, action_shape, False, None)
        self.log_sigma = Layer(h, action_shape, False, None)


    def call(self, x):
        
        x = self.seq(x)

        mean = self.mean(x)

        log_sigma = self.log_sigma(x) 
        log_sigma = tf.clip_by_value(log_sigma, LOG_SIG_MIN, LOG_SIG_MAX)

        return mean, log_sigma

 


class QValueNet(tf.keras.Model):

    def __init__(self, args=critic_args):

 
        h = args.hidden_dim
        ln = args.layer_norm
        afn = args.afn
        res = args.residual
        drop = args.dropout
        q_dim = args.q_value_dim + 1 


        super().__init__()
 
        tgt_dim, obs_dim = observation_shape
         
        self.seq = tf.keras.Sequential([
            Layer(obs_dim + tgt_dim + action_shape, h, ln, afn, res, drop),
            Layer(h, h, ln, afn, res, drop),
            Layer(h, h, ln, afn, res, drop),
        ])

        self.q_value = Layer(h, q_dim, False, None)
 

    def call(self, obs, act):
        
        x = tf.concat([obs, act], axis=-1)
        x = self.seq(x)
        return self.q_value(x)







def create_nets():
 

    policy_net = PolicyNet()
 
    q_net_1 = QValueNet()
    q_net_2 = QValueNet()
    target_q_net_1 = QValueNet()
    target_q_net_2 = QValueNet()
 
    target_q_net_1.set_weights(q_net_1.get_weights())
    target_q_net_2.set_weights(q_net_2.get_weights())

    policy_optim = tf.keras.optimizers.Adam(learning_rate=actor_args.lr)  
    q1_optim = tf.keras.optimizers.Adam(learning_rate=critic_args.lr)  
    q2_optim = tf.keras.optimizers.Adam(learning_rate=critic_args.lr) 

    return policy_net, q_net_1, q_net_2, target_q_net_1, \
                    target_q_net_2, policy_optim, q1_optim, q2_optim


  
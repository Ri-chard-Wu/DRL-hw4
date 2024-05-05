
from env_wrappers import make_train_env, make_test_env
from utils import AttrDict
 
env_num = 32
segment_len = 10
 
train_env = make_train_env()
test_env make_test_env()
  
q_value_dim = 6
q_weights = [2.0, 1.0, 1.0, 1.0, 1.0, 1.0]

agent = SAC()
segment_sampler = SegmentSampler(agent, train_env, segment_len, q_weights)
experience_replay = PrioritizedExperienceReplay(segment_len, q_value_dim)

trainer = Trainer(env_num, test_env, segment_sampler, agent, experience_replay)
trainer.train()

train_env.close()
test_env.close()




from env_wrappers import make_train_env, make_test_env
from sac import SAC
from segment_sampler import SegmentSampler
from experience_replay import PrioritizedExperienceReplay
from trainer import Trainer

train_env = make_train_env()
test_env = make_test_env()
   
agent = SAC()
segment_sampler = SegmentSampler(agent, train_env)
experience_replay = PrioritizedExperienceReplay()

trainer = Trainer(test_env, segment_sampler, agent, experience_replay)
trainer.train()

train_env.close()
test_env.close()

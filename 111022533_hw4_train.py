
# from env_wrappers import make_train_env
from sac import SAC
from segment_sampler import SegmentSampler
from experience_replay import PrioritizedExperienceReplay
from trainer import Trainer

# train_env = make_train_env() 

agent = SAC()
segment_sampler = SegmentSampler(agent)
experience_replay = PrioritizedExperienceReplay()

trainer = Trainer(segment_sampler, agent, experience_replay)
trainer.train()

# train_env.close()


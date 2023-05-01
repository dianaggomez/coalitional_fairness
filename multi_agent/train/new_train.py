import gym
import gym_marl_hlc
import numpy as np
import torch

from stable_baselines3 import PPO
from callback_eval import TimeCallback, StopTrainingOnNoModelImprovement
import random
# import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

print("Is device available:",  torch.cuda.is_available())
print("Cuda version:",  torch.version.cuda)

random.seed(0)
np.random.seed(seed=0)
torch.manual_seed(0)

eval_env = gym.make("gym_marl_hlc/marl_hlc-v0")

time_out_step = 10 # max number of time_steps in an episode
n_samples = 2000  #52 # number of trajectories we would like to sample
n_step = time_out_step * n_samples
epochs = 20 #10 #30
total_time= n_step*epochs
batch_size = 64 #400 #104


no_improvement_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals = 3, min_evals = 5, verbose = 1)

evaluate_callback= TimeCallback(eval_env, 
                                eval_freq=n_step, #520, 
                                verbose=1,
                                callback_after_eval=no_improvement_callback, 
                                best_model_save_path="./models/alpha_0/best_success/",
                                n_eval_episodes=4000)
model = PPO("MlpPolicy",
            eval_env, verbose=1, 
            tensorboard_log = "./models/alpha_0",  
            device = "cuda:1", 
            batch_size=batch_size)

model.learn(total_timesteps=total_time, callback= evaluate_callback)



import gym_marl_hlc
import gym
import numpy as np
import tensorflow as tf
from stable_baselines3 import PPO

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['SDL_AUDIODRIVER'] = 'dsp'

np.random.seed(seed=0)
tf.random.set_random_seed(0)

#environment
env = gym.make("gym_marl_hlc/marl_hlc-v0")

model = PPO.load("/home/diana/journal/multi_agent/train/models/alpha_1/best_success/best_model.zip")
obs = env.reset()
print("First Queue: ", env.original_queue)
time_step = 0.2

baseline = 122.01
exit_time = []
# successes_data = np.load("/home/diana/journal/train/final_queues.npy")
# successes = len(successes_data)
# print("Number of Successes: ", successes)

queue_recorded = []
successful_queues = []
# for i in range(successes):
for i in range(4004):
    if i%25 == 0: 
        print(i)
    done = False
    # print(type(env.original_queue))
    while done == False:
        # print("###########################################################")
        # print("Action: ", action[0])
        action = model.predict(obs)
        obs, r, done, info = env.step(action[0])  
        # print(info)
        # print(obs)
        # print("Timesteps", env.LowLevelControllerEnv.time)
        # print("###########################################################")
    # exit_time.append(np.append(env.original_queue, [env.queue_config, max(info["AVs_time"], info["humans_time"])]))
    # queue_recorded.append(env.LowLevelControllerEnv.original_queue)
    # print(exit_time)
    
    # # print(info)
    # if info["AVs_cleared_queue"]:
    if info["Humans_cleared"]: #bc we want AV_2 which is labeled as humans
        successful_queues.append(np.array(env.original_queue))

    obs = env.reset()

successful_queues = np.array(successful_queues)
print("queues recorded: ", len(successful_queues))
np.save("/home/diana/journal/multi_agent/train/models/alpha_1/successful_queues_1.npy", successful_queues)

# exit_time = np.array(exit_time)
# np.save("/home/diana/journal/train/models/alpha_0/exit_time_8.npy", exit_time)

# queue_recorded = np.array(queue_recorded)
# np.save("/home/diana/journal/train/models/alpha_0/queue_recorded_8.npy", queue_recorded)
    
import gym
from gym.envs.registration import register, registry
# from gym_hlc.envs.hlc import HighLevelControllerEnv

# env_name = "hlc-v0"
env_name = 'gym_marl_hlc/marl_hlc-v0'
# envs=[]

env_dict = gym.envs.registration.registry.env_specs.copy()

for env in env_dict:
    if env_name in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]
        
register(id=env_name, 
         entry_point='gym_marl_hlc.marl_envs:MARLHighLevelControllerEnv')

    
if __name__ == '__main__':
    # Test purpose only
    import gym

    env = gym.make("gym_marl_hlc/marl_hlc-v0")
    env.reset()
    env.close()

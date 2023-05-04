from copo.callbacks import MultiAgentDrivingCallbacks, MultiAgentHLCCallbacks
from copo.algo_ippo.ippo import IPPOTrainer
from copo.train.train import train
from copo.train.utils import get_train_parser
from copo.utils import get_rllib_compatible_env
from ray import tune

import sys
sys.path.insert(0, '/home/diana/coalitional_fairness/coalitional_fairness') #place your own path
from multi_agent.gym_marl_hlc.marl_envs import MARLHighLevelControllerEnv

if __name__ == "__main__":
    args = get_train_parser().parse_args()
    exp_name = args.exp_name or "TEST"

    # test env works
    env = MARLHighLevelControllerEnv(config=None)
    obs = env.reset()
    print(obs)
  
    # Setup config
    # stop = int(100_0000)
    stop = int(1000)

    config = dict(
        env=get_rllib_compatible_env(MARLHighLevelControllerEnv),
        env_config=dict(),
        # ===== Resource =====
        # So we need 2 CPUs per trial, 0.25 GPU per trial!
        num_gpus=0.5 if args.num_gpus != 0 else 0,
    ) 


    # Launch training
    train(
        IPPOTrainer,
        exp_name=exp_name,
        keep_checkpoints_num=3,
        stop=stop,
        config=config,
        num_gpus=args.num_gpus,
        num_seeds=1,
        test_mode=args.test,
        custom_callback=MultiAgentHLCCallbacks,

        # fail_fast='raise',
        local_mode=True, #this allows us to print/debug
    )

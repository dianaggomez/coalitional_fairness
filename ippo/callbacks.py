from collections import defaultdict
from typing import Dict

import numpy as np
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy


class MultiAgentHLCCallbacks(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.episode_reward = 0.0
        self.last_num_timesteps = 0
        self.eps = 0
        
        self.last_mean_av_change= -np.inf
        self.last_best_mean_av_change = -np.inf
        
        self.last_best_mean_success = -np.inf
        self.last_mean_success = -np.inf
        
        self._is_av_time = []
        self._is_human_time = []
        self.r_f = []
        
        self._is_success_buffer = []
        
    def on_episode_start(
        self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: MultiAgentEpisode,
        env_index: int, **kwargs
    ):
        # define all the parameters to keep track of during evaluation
        episode.user_data["time_change"] = defaultdict(list)
        episode.user_data["fairness_reward"] = defaultdict(list)
        episode.user_data["success_rate"] = defaultdict(list)
        episode.user_data["episode_length"] = defaultdict(list)
        episode.user_data["episode_reward"] = defaultdict(list)
        

    def on_episode_step(
        self, *, worker: RolloutWorker, base_env: BaseEnv, episode: MultiAgentEpisode, env_index: int, **kwargs
    ):
        
        #active_keys = list(base_env.envs[env_index].vehicles.keys())
        
        # equivilantly we can look at the self.dones and see which agents are still active
        active_keys = []

        for agent_id in base_env.envs[env_index]._agent_ids:
            if base_env.envs[env_index].dones[agent_id] != True:
                active_keys.append(agent_id)

        # The agent_rewards dict contains all agents' reward, not only the active agent!
        # active_keys = [k for k, _ in episode.agent_rewards.keys()]

        for agent_id in active_keys:
            k = agent_id
            info = episode.last_info_for(k)
            if info:
                #TODO: check if this is necessary
                if "step_reward" not in info:
                    continue
                #TODO: need to add time into the info for each agent
                episode.user_data["time_change"][k].append(info["time_change"])
                # OR we can continue to check if the agent is done to log 
                

    def on_episode_end(
        self, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: MultiAgentEpisode,
        **kwargs
    ):
        keys = [k for k, _ in episode.agent_rewards.keys()]
        # arrive_dest_list = []
        # crash_list = []
        # out_of_road_list = []
        # max_step_rate_list = []
        
        # at the moment the info is the same for both agents
        for k in keys:
            info = episode.last_info_for(k)
            maybe_is_success = info.get("AVs_cleared_queue")
            maybe_av_time = info.get("AVs_time")
            maybe_human_time = info.get("humans_time")
            fair_reward = info.get("fair_reward")
            
            # TODO: Need to make sure that this is for 2 agents
            if maybe_is_success is not None:
                try:
                    self._is_success_buffer.append(maybe_is_success)
                    if maybe_is_success:
                        # self._is_success_buffer.append(maybe_is_success)
                        self._is_av_time.append(maybe_av_time)
                        self._is_human_time.append(maybe_human_time)
                        # log the reward associated with fairness
                        self.r_f.append(fair_reward)
                    # self._is_queue.append(maybe_queue)
                except:
                    print("Dones with errors: ", maybe_av_time, maybe_human_time)
                    
        episode.custom_metrics["success_rate"] = np.mean(self._is_success_buffer)
                 
        # for k in keys:
        #     info = episode.last_info_for(k)
            
        #     arrive_dest = info.get("arrive_dest", False)
        #     crash = info.get("crash", False)
        #     out_of_road = info.get("out_of_road", False)
        #     max_step_rate = not (arrive_dest or crash or out_of_road)
        #     arrive_dest_list.append(arrive_dest)
        #     crash_list.append(crash)
        #     out_of_road_list.append(out_of_road)
        #     max_step_rate_list.append(max_step_rate)
        # episode.custom_metrics["success_rate"] = np.mean(arrive_dest_list)
        # episode.custom_metrics["crash_rate"] = np.mean(crash_list)
        # episode.custom_metrics["out_of_road_rate"] = np.mean(out_of_road_list)
        # episode.custom_metrics["max_step_rate"] = np.mean(max_step_rate_list)
        # for info_k, info_dict in episode.user_data.items():
        #     self._add_item(episode, info_k, [vv for v in info_dict.values() for vv in v])
        # agent_cost_list = [sum(episode_costs) for episode_costs in episode.user_data["cost"].values()]
        # episode.custom_metrics["episode_cost"] = np.mean(agent_cost_list)
        # episode.custom_metrics["episode_cost_worst_agent"] = np.min(agent_cost_list)
        # episode.custom_metrics["episode_cost_best_agent"] = np.max(agent_cost_list)
        # episode.custom_metrics["environment_cost_total"] = np.sum(agent_cost_list)
        # episode.custom_metrics["num_active_agents"] = len(agent_cost_list)
        episode.custom_metrics["episode_length"] = np.mean(
            [ep_len[-1] for ep_len in episode.user_data["episode_length"].values()]
        )
        episode.custom_metrics["episode_reward"] = np.mean(
            [ep_r[-1] for ep_r in episode.user_data["episode_reward"].values()]
        )
        # episode.custom_metrics["environment_reward_total"] = np.sum(
        #     [ep_r[-1] for ep_r in episode.user_data["episode_reward"].values()]
        # )

    def _add_item(self, episode, name, value_list):
        episode.custom_metrics["{}_max".format(name)] = float(np.max(value_list))
        episode.custom_metrics["{}_mean".format(name)] = float(np.mean(value_list))
        episode.custom_metrics["{}_min".format(name)] = float(np.min(value_list))

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        result["success"] = np.nan
        result["crash"] = np.nan
        result["out"] = np.nan
        result["max_step"] = np.nan
        result["length"] = result["episode_len_mean"]
        if "success_rate_mean" in result["custom_metrics"]:
            result["success"] = result["custom_metrics"]["success_rate_mean"]
            result["crash"] = result["custom_metrics"]["crash_rate_mean"]
            result["out"] = result["custom_metrics"]["out_of_road_rate_mean"]
            result["max_step"] = result["custom_metrics"]["max_step_rate_mean"]
        result["cost"] = np.nan
        if "episode_cost_mean" in result["custom_metrics"]:
            result["cost"] = result["custom_metrics"]["episode_cost_mean"]

        # present the agent-averaged reward.
        result["raw_episode_reward_mean"] = result["episode_reward_mean"]
        result["episode_reward_mean"] = np.mean(list(result["policy_reward_mean"].values()))
        # result["environment_reward_total"] = np.sum(list(result["policy_reward_mean"].values()))
    
    
    
    
class MultiAgentDrivingCallbacks(DefaultCallbacks):
    def on_episode_start(
        self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: MultiAgentEpisode,
        env_index: int, **kwargs
    ):
        episode.user_data["velocity"] = defaultdict(list)
        episode.user_data["steering"] = defaultdict(list)
        episode.user_data["step_reward"] = defaultdict(list)
        episode.user_data["acceleration"] = defaultdict(list)
        episode.user_data["cost"] = defaultdict(list)
        episode.user_data["episode_length"] = defaultdict(list)
        episode.user_data["episode_reward"] = defaultdict(list)
        episode.user_data["num_neighbours"] = defaultdict(list)

    def on_episode_step(
        self, *, worker: RolloutWorker, base_env: BaseEnv, episode: MultiAgentEpisode, env_index: int, **kwargs
    ):
        active_keys = list(base_env.envs[env_index].vehicles.keys())

        # The agent_rewards dict contains all agents' reward, not only the active agent!
        # active_keys = [k for k, _ in episode.agent_rewards.keys()]

        for agent_id in active_keys:
            k = agent_id
            info = episode.last_info_for(k)
            if info:
                if "step_reward" not in info:
                    continue
                episode.user_data["velocity"][k].append(info["velocity"])
                episode.user_data["steering"][k].append(info["steering"])
                episode.user_data["step_reward"][k].append(info["step_reward"])
                episode.user_data["acceleration"][k].append(info["acceleration"])
                episode.user_data["cost"][k].append(info["cost"])
                episode.user_data["episode_length"][k].append(info["episode_length"])
                episode.user_data["episode_reward"][k].append(info["episode_reward"])
                episode.user_data["num_neighbours"][k].append(len(info.get("neighbours", [])))

    def on_episode_end(
        self, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: MultiAgentEpisode,
        **kwargs
    ):
        keys = [k for k, _ in episode.agent_rewards.keys()]
        arrive_dest_list = []
        crash_list = []
        out_of_road_list = []
        max_step_rate_list = []
        for k in keys:
            info = episode.last_info_for(k)
            arrive_dest = info.get("arrive_dest", False)
            crash = info.get("crash", False)
            out_of_road = info.get("out_of_road", False)
            max_step_rate = not (arrive_dest or crash or out_of_road)
            arrive_dest_list.append(arrive_dest)
            crash_list.append(crash)
            out_of_road_list.append(out_of_road)
            max_step_rate_list.append(max_step_rate)
        episode.custom_metrics["success_rate"] = np.mean(arrive_dest_list)
        episode.custom_metrics["crash_rate"] = np.mean(crash_list)
        episode.custom_metrics["out_of_road_rate"] = np.mean(out_of_road_list)
        episode.custom_metrics["max_step_rate"] = np.mean(max_step_rate_list)
        for info_k, info_dict in episode.user_data.items():
            self._add_item(episode, info_k, [vv for v in info_dict.values() for vv in v])
        agent_cost_list = [sum(episode_costs) for episode_costs in episode.user_data["cost"].values()]
        episode.custom_metrics["episode_cost"] = np.mean(agent_cost_list)
        episode.custom_metrics["episode_cost_worst_agent"] = np.min(agent_cost_list)
        episode.custom_metrics["episode_cost_best_agent"] = np.max(agent_cost_list)
        episode.custom_metrics["environment_cost_total"] = np.sum(agent_cost_list)
        episode.custom_metrics["num_active_agents"] = len(agent_cost_list)
        episode.custom_metrics["episode_length"] = np.mean(
            [ep_len[-1] for ep_len in episode.user_data["episode_length"].values()]
        )
        episode.custom_metrics["episode_reward"] = np.mean(
            [ep_r[-1] for ep_r in episode.user_data["episode_reward"].values()]
        )
        episode.custom_metrics["environment_reward_total"] = np.sum(
            [ep_r[-1] for ep_r in episode.user_data["episode_reward"].values()]
        )

    def _add_item(self, episode, name, value_list):
        episode.custom_metrics["{}_max".format(name)] = float(np.max(value_list))
        episode.custom_metrics["{}_mean".format(name)] = float(np.mean(value_list))
        episode.custom_metrics["{}_min".format(name)] = float(np.min(value_list))

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        result["success"] = np.nan
        result["crash"] = np.nan
        result["out"] = np.nan
        result["max_step"] = np.nan
        result["length"] = result["episode_len_mean"]
        if "success_rate_mean" in result["custom_metrics"]:
            result["success"] = result["custom_metrics"]["success_rate_mean"]
            result["crash"] = result["custom_metrics"]["crash_rate_mean"]
            result["out"] = result["custom_metrics"]["out_of_road_rate_mean"]
            result["max_step"] = result["custom_metrics"]["max_step_rate_mean"]
        result["cost"] = np.nan
        if "episode_cost_mean" in result["custom_metrics"]:
            result["cost"] = result["custom_metrics"]["episode_cost_mean"]

        # present the agent-averaged reward.
        result["raw_episode_reward_mean"] = result["episode_reward_mean"]
        result["episode_reward_mean"] = np.mean(list(result["policy_reward_mean"].values()))
        # result["environment_reward_total"] = np.sum(list(result["policy_reward_mean"].values()))

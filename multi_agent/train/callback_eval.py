
import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

import gym
import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

class StopTrainingOnNoModelImprovement(BaseCallback):
    """
    Stop the training early if there is no new best model (new best mean reward) after more than N consecutive evaluations.
    It is possible to define a minimum number of evaluations before start to count evaluations without improvement.
    It must be used with the ``EvalCallback``.
    :param max_no_improvement_evals: Maximum number of consecutive evaluations without a new best model.
    :param min_evals: Number of evaluations before start to count evaluations without improvements.
    :param verbose: Verbosity level: 0 for no output, 1 for indicating when training ended because no new best model
    """

    def __init__(self, max_no_improvement_evals: int, min_evals: int = 0, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.max_no_improvement_evals = max_no_improvement_evals
        self.min_evals = min_evals
        self.last_best_mean_success = -np.inf
        self.last_best_mean_av_change = np.inf
        self.no_improvement_evals = 0

    def _on_step(self) -> bool:
        assert self.parent is not None, "``StopTrainingOnNoModelImprovement`` callback must be used with an ``EvalCallback``"

        continue_training = True

        print("Parent Last Mean: ", self.parent.last_best_mean_success)
        # print("Child Last Mean: ", self.last_best_mean_success)
        
        if self.n_calls > self.min_evals:
            if self.parent.last_best_mean_success > self.last_best_mean_success: 
                self.no_improvement_evals = 0
            # elif self.parent.last_best_mean_success == self.last_best_mean_success:
            #     if self.parent.last_best_mean_av_change < self.last_best_mean_av_change:
            #         self.no_improvement_evals = 0
            else:
                self.no_improvement_evals += 1
                if self.no_improvement_evals > self.max_no_improvement_evals:
                    continue_training = False

        self.last_best_mean_success = self.parent.last_best_mean_success
        # self.last_best_mean_av_change = self.parent.last_best_mean_av_change
        
        print("No Improvement Evals: " , self.no_improvement_evals)
        
        if self.verbose >= 1 and not continue_training:
            print(
                f"Stopping training because there was no new best model in the last {self.no_improvement_evals:d} evaluations"
            )

        return continue_training

class TimeCallback(EvalCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose:
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """
    def __init__(self, eval_env, eval_freq, verbose, callback_after_eval, best_model_save_path, n_eval_episodes):
        super(TimeCallback, self).__init__(eval_env=eval_env,
                                           eval_freq=eval_freq,
                                           verbose=verbose, 
                                           callback_after_eval=callback_after_eval, 
                                           best_model_save_path=best_model_save_path,
                                           n_eval_episodes = n_eval_episodes)
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


    
    def _log_time_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
          
        info = locals_["info"]
        # print(info)
        
        if locals_["dones"]:
            # print(self._is_success_buffer)
            maybe_is_success = info.get("AVs_cleared_queue")
            maybe_av_time = info.get("AVs_time")
            maybe_human_time = info.get("humans_time")
            fair_reward = info.get("fair_reward")
            
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


    def _on_step(self) -> bool:
        mean_av_change = 0

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    )
            
            # Reset success rate buffer
            self._is_success_buffer = []
            self._is_av_time = []
            self._is_human_time = []
            self.r_f = []
            
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes, #evaluate all queues
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_time_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            # print("How many successes: ", len(self._is_success_buffer))
            
            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                self.logger.record("eval/success_rate", success_rate)
                
            if len(self._is_av_time) > 0:
                # Keep track on of the min, mean, max change
                min_av_change = np.min(self._is_av_time)
                max_av_change = np.max(self._is_av_time)
                mean_av_change = np.mean(self._is_av_time)
                # success_rate = np.mean(self._is_success_buffer)
                
                # Info about fairness reward
                min_fair_reward = np.min(self.r_f)
                max_fair_reward = np.max(self.r_f)
                mean_fair_reward = np.mean(self.r_f)
                
                if self.verbose > 0:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                    print(f"Min AV Time change: {min_av_change:.2f}")
                    print(f"Max AV Time change: {max_av_change:.2f}")
                    print(f"Mean AV Time change: {mean_av_change:.2f}")
                    

                self.logger.record("eval/min_av_change", min_av_change)
                self.logger.record("eval/max_av_change", max_av_change)
                self.logger.record("eval/mean_av_change", mean_av_change)
                
                self.logger.record("eval/min_fair_reward", min_fair_reward)
                self.logger.record("eval/max_fair_reward", max_fair_reward)
                self.logger.record("eval/mean_fair_reward", mean_fair_reward)
                # self.logger.record("eval/success_rate", success_rate)
                

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)
            
            self.last_mean_av_change = mean_av_change
            self.last_mean_success = success_rate
            
            if success_rate > self.last_best_mean_success:
                if self.verbose > 0:
                    print("New best mean success rate!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.last_best_mean_success = success_rate
                self.last_best_mean_av_change = mean_av_change
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()
        
        return True

# class TimeCallback(EvalCallback):
#     """
#     Callback for evaluating an agent.

#     .. warning::

#       When using multiple environments, each call to  ``env.step()``
#       will effectively correspond to ``n_envs`` steps.
#       To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

#     :param eval_env: The environment used for initialization
#     :param callback_on_new_best: Callback to trigger
#         when there is a new best model according to the ``mean_reward``
#     :param n_eval_episodes: The number of episodes to test the agent
#     :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
#     :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
#         will be saved. It will be updated at each evaluation.
#     :param best_model_save_path: Path to a folder where the best model
#         according to performance on the eval env will be saved.
#     :param deterministic: Whether the evaluation should
#         use a stochastic or deterministic actions.
#     :param render: Whether to render or not the environment during evaluation
#     :param verbose:
#     :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
#         wrapped with a Monitor wrapper)
#     """
#     def __init__(self, eval_env, eval_freq, verbose=1):
#         super(TimeCallback, self).__init__(eval_env=eval_env,eval_freq=eval_freq,verbose=verbose)
#         self.episode_reward = 0.0
#         self.last_num_timesteps = 0
#         self.eps = 0
#         self.best_mean_group_change= -np.inf
#         self.last_mean_group_change = -np.inf
        
#         self._is_avs_time = []
#         self._is_humans_time = []
#         self._is_group_time = []
#         self.eval_env.training = False

    
#     def time_data(self):
#         '''
#         Time for t_{\pi} t_{SR} and t_{group}
#         '''
#         baseline = 121.01 #seconds
#         time_to_exit = {}
#         obs =  self.eval_env.reset()
        
#         # print("First Queue: ", env.queue)
#         time_step = 0.2
#         counts = 0

#         self.eval_env.training = False
#         while len(time_to_exit) < 10: 
#             counts +=1
#             action, _states = self.model.predict(obs)
#             obs, rewards, dones, info = self.eval_env.step(action)
#             if dones or counts > 25:
#                 if counts <= 25:
#                     counts = 0
#                     time_to_exit[self.eval_env.queue] = [self.eval_env.AVs_time*time_step, 
#                                                     self.eval_env.Humans_time*time_step, 
#                                                     max(self.eval_env.AVs_time, self.eval_env.Humans_time)*time_step]
#                 obs = self.eval_env.reset()
                
#         t_group = np.array(list(time_to_exit.values()))[:, 2] - baseline
            
#         # save the data to npy
#         return t_group
    
#     def maths(self, time):
#         max_change = np.max(np.abs(time))
#         min_change = np.min(np.abs(time))
#         mean_change = np.mean(np.abs(time))
#         return max_change, min_change, mean_change
    
        
#     def _on_step(self) -> bool:

#         if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
#             # Sync training and eval env if there is VecNormalize
#             if self.model.get_vec_normalize_env() is not None:
#                 try:
#                     sync_envs_normalization(self.training_env, self.eval_env)
#                 except AttributeError:
#                     raise AssertionError(
#                         "Training and eval env are not wrapped the same way, "
#                         "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
#                         "and warning above."
#                     )
                    
#             print()       
#             t_group = self.time_data()
#             max_t_group, min_t_group, mean_t_group = self.maths(t_group)
#             print( mean_t_group)   
#             # if self.verbose > 0:
#             #     print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
#             #     print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            
#             # # Add to current Logger
#             # self.logger.record("eval/mean_t_group", float(max_t_group))
                
                

#             # # Dump log so the evaluation results are printed with the correct timestep
#             # self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
#             # self.logger.dump(self.num_timesteps)

#             # if mean_reward > self.best_mean_reward:
#             #     if self.verbose > 0:
#             #         print("New best mean reward!")
#             #     if self.best_model_save_path is not None:
#             #         self.model.save(os.path.join(self.best_model_save_path, "best_model"))
#             #     self.best_mean_reward = mean_reward
#             #     # Trigger callback if needed
#             #     if self.callback is not None:
#             #         return self._on_event()

#         return True





    
class DrivingCallback(EvalCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose:
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]
        
        if locals_["dones"]:
            # print(self._is_success_buffer)
            maybe_is_success = info.get("arrive_dest")
            maybe_is_crash = info.get("crash")
            maybe_is_oor = info.get("out_of_road")
            maybe_is_max_step = info.get("max_step")
            
            if maybe_is_success is not None:
                try:
                    self._is_success_buffer.append(maybe_is_success)
                    self._is_oor_buffer.append(maybe_is_oor)
                    self._is_crash_buffer.append(maybe_is_crash)
                    self._is_max_step_buffer.append(maybe_is_max_step)
                except:
                    print("Dones with errors: ", maybe_is_oor, maybe_is_crash, maybe_is_max_step)
                        
    
    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    )

            # Reset success rate buffer
            self._is_success_buffer = []
            self._is_crash_buffer = []
            self._is_oor_buffer = []
            self._is_max_step_buffer = []
            
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                crash_rate = np.mean(self._is_crash_buffer)
                oor_rate = np.mean(self._is_oor_buffer)
                max_time_rate = np.mean(self._is_max_step_buffer)
                if self.verbose > 0:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                    print(f"Crash rate: {100 * crash_rate:.2f}%")
                    print(f"Out of Road rate: {100 * oor_rate:.2f}%")
                    print(f"Max time rate: {100 * max_time_rate:.2f}%")
                    

                self.logger.record("eval/success_rate", success_rate)
                self.logger.record("eval/crash_rate", crash_rate)
                self.logger.record("eval/oor_rate", oor_rate)
                self.logger.record("eval/max_time_rate", max_time_rate)
                
                

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        return True





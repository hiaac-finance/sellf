import time
import os
from copy import deepcopy
from typing import Any, Callable, List, Optional, Tuple, Union, Dict

import csv
import gym
import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback


class BenefitDelta:
    """
    Used to compute the benefit \delta(x,s) for each credit score x=c and group s=g
    using p(x'|x,d,do(s)), which is computed using the credit drift and repayment probabilities.
    """
    def __init__(self, config):
        self.DRIFT_PROBS = config.DRIFT_PROBS
        self.DELAYED_IMPACT_SUCCESS_PROBS = config.DELAYED_IMPACT_SUCCESS_PROBS
        self.QUAL_CHANGE = config.QUAL_CHANGE

    def _prob_c(self, c, g):
        return self.DRIFT_PROBS[g][c+1]

    def _prob_y_condx(self, x, y):
        if y == 0:
            return 1 - self.DELAYED_IMPACT_SUCCESS_PROBS[x]
        else:
            return self.DELAYED_IMPACT_SUCCESS_PROBS[x]
        
    def _condition_1(self, x, a, g):
        if a == 1:
            return self._prob_y_condx(x, y=1) * self._prob_c(1, g)
        else:
            return 0.0

    def _condition_2(self, x, a, g):
        if a == 1:
            return self._prob_y_condx(x, y=1) * self._prob_c(0, g)
        else:
            return self._prob_y_condx(x, y=1) * self._prob_c(1, g) + self._prob_y_condx(x, y=0) * self._prob_c(1, g)

    def _condition_3(self, x, a, g):
        if a == 1:
            return self._prob_y_condx(x, y=1) * self._prob_c(-1, g) + self._prob_y_condx(x, y=0) * self._prob_c(1, g)
        else:
            return self._prob_y_condx(x, y=1) * self._prob_c(0, g) +self._prob_y_condx(x, y=0) * self._prob_c(0, g)

    def _condition_4(self, x, a, g):
        if a == 1:
            return self._prob_y_condx(x, y=0) * self._prob_c(0, g) 
        else:
            return self._prob_y_condx(x, y=1) * self._prob_c(-1, g) + self._prob_y_condx(x, y=0) * self._prob_c(-1, g)

    def _condition_5(self, x, a, g):
        if a == 1:
            return self._prob_y_condx(x, y=0) * self._prob_c(-1, g)
        else:
            return 0.0

    def _get_px_prime_cond_xd(self, x, a, x_prime, max_x, min_x, g):
        if x == max_x:
            if x_prime == max_x:
                return self._condition_1(x, a, g) + self._condition_2(x, a, g) + self._condition_3(x, a, g)
        elif x == min_x:
            if x_prime == min_x:
                return self._condition_3(x, a, g) + self._condition_4(x, a, g) + self._condition_5(x, a, g)
        else:
            if x == max_x - 1:
                if x_prime >= x:
                    return self._condition_1(x, a, g) + self._condition_2(x, a, g)

            if x == min_x + 1:
                if x_prime <= x:
                    return self._condition_4(x, a, g) + self._condition_5(x, a, g)
                
        if x_prime == x:
            # print('condition_3')
            return self._condition_3(x, a, g)
        elif x_prime == x - 1:
            # print('condition_4')
            return self._condition_4(x, a, g)
        elif x_prime == x - 2:
            # print('condition_5')
            return self._condition_5(x, a, g)
        elif x_prime == x + 1:
            # print('condition_1')
            return self._condition_2(x, a, g)
        elif x_prime == x + 2:
            # print('condition_2')
            return self._condition_1(x, a, g)
        
    def _compute_benefit_delta(self, x: int, g) -> float:
        max_x = len(self.DELAYED_IMPACT_SUCCESS_PROBS) - 1
        min_x = 0
        x_primes = None
        if x == max_x:
            x_primes = [max_x, max_x - 1, max_x - 2]
        elif x == max_x - 1:
            x_primes = [x+1, x, x-1, x-2]
        elif x == min_x:
            x_primes = [min_x, min_x + 1, min_x + 2]
        elif x == min_x + 1:
            x_primes = [x-1, x, x+1, x+2]
        else:
            x_primes = [x-2, x-1, x, x+1, x+2]

        benefit_delta = 0.0
        for x_prime in x_primes:
            temp_delta = self._get_px_prime_cond_xd(x, 1, x_prime, max_x, min_x, g) - self._get_px_prime_cond_xd(x, 0, x_prime, max_x, min_x, g)
            benefit_delta += temp_delta * self.QUAL_CHANGE(x, x_prime)

        return benefit_delta

    def gather_benefit_deltas(self) -> Dict[int, float]:
        deltas = {}
        for g in range(len(self.DRIFT_PROBS)):
            for i in range(len(self.DELAYED_IMPACT_SUCCESS_PROBS)):
                if g not in deltas:
                    deltas[g] = {}
                deltas[g][i] = self._compute_benefit_delta(i,g)

        return deltas
    

class QDummyVecEnv(DummyVecEnv):
    '''
    Extends the DummyVecEnv class to return the 'g' reward as well
    
    '''
    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        super(QDummyVecEnv, self).__init__(env_fns)
        self.buf_g_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_g_pi0_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_pi0_actions = np.zeros((self.num_envs,), dtype=np.int16)
        self.buf_pi0_next_x = np.zeros((self.num_envs,), dtype=np.int16)

    def step_wait(self) -> VecEnvStepReturn:
        for env_idx in range(self.num_envs):
            obs, self.buf_rews[env_idx], self.buf_dones[env_idx], self.buf_infos[env_idx], self.buf_g_rews[env_idx], self.buf_g_pi0_rews[env_idx], self.buf_pi0_actions[env_idx], self.buf_pi0_next_x[env_idx] = self.envs[env_idx].step(
                self.actions[env_idx]
            )
            # breakpoint()
            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]["terminal_observation"] = obs
                obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos), np.copy(self.buf_g_rews), np.copy(self.buf_g_pi0_rews), np.copy(self.buf_pi0_actions), np.copy(self.buf_pi0_next_x))
    

class QMonitor(Monitor):
    """
    A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.

    :param env: The environment
    :param filename: the location to save a log file, can be None for no log
    :param allow_early_resets: allows the reset of the environment before it is done
    :param reset_keywords: extra keywords for the reset call,
        if extra parameters are needed at reset
    :param info_keywords: extra information to log, from the information return of env.step()
    """

    EXT = "monitor.csv"

    def __init__(
        self,
        env: gym.Env,
        filename: Optional[str] = None,
        allow_early_resets: bool = True,
        reset_keywords: Tuple[str, ...] = (),
        info_keywords: Tuple[str, ...] = (),
    ):
        super(QMonitor, self).__init__(
            env=env,
            filename=filename,
            allow_early_resets=allow_early_resets,
            reset_keywords=reset_keywords,
            info_keywords=info_keywords,
            )

        self.g_rewards = None
        self.g_pi0_rewards = None
    
    # override the step method to reset the 'g' reward as well
    def reset(self, **kwargs) -> GymObs:
        """
        Calls the Gym environment reset. Can only be called if the environment is over, or if allow_early_resets is True

        :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
        :return: the first observation of the environment
        """
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError(
                "Tried to reset an environment before done. If you want to allow early resets, "
                "wrap your env with Monitor(env, path, allow_early_resets=True)"
            )
        self.rewards = []
        self.g0_rewards = []
        self.g1_rewards = []
        self.g0_pi0_rewards = []
        self.g1_pi0_rewards = []
        self.needs_reset = False
        for key in self.reset_keywords:
            value = kwargs.get(key)
            if value is None:
                raise ValueError(f"Expected you to pass keyword argument {key} into reset")
            self.current_reset_info[key] = value
        return self.env.reset(**kwargs)

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        """
        Step the environment with the given action

        :param action: the action
        :return: observation, reward, done, information
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        
        is_g0 = True if np.argmax(self.env.env.state.group) == 0 else False

        observation, reward, done, info, g_reward, g_pi0_reward, action_pi0, next_x_pi0 = self.env.step(action)
        # breakpoint()
        self.rewards.append(reward)
        # import pdb; pdb.set_trace()
        # if observation[7].astype(int) == 1:
        if is_g0:
            self.g0_rewards.append(g_reward)
            self.g0_pi0_rewards.append(g_pi0_reward)
        else:
            self.g1_rewards.append(g_reward)
            self.g1_pi0_rewards.append(g_pi0_reward)

        # self.g_pi0_rewards.append(g_pi0_reward)
        if done:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            # print(sum(self.g0_rewards))
            # print(sum(self.g1_rewards))
            ep_g0_rew = sum(self.g0_rewards)
            ep_g1_rew = sum(self.g1_rewards)
            ep_g0pi0_rew = sum(self.g0_pi0_rewards)
            ep_g1pi0_rew = sum(self.g1_pi0_rewards)
            # ep_g_rew = sum(self.g_rewards)
            ep_len = len(self.rewards)
            ep_info = {"r": round(ep_rew, 6),'g0r': round(ep_g0_rew), 'g1r': round(ep_g1_rew), 'g0_pi0_r': round(ep_g0pi0_rew), 'g1_pi0_r': round(ep_g1pi0_rew), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_returns.append(ep_rew)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(ep_info)
            info["episode"] = ep_info
        # if self.total_steps == 4:
        #     breakpoint()
        self.total_steps += 1
        
        return observation, reward, done, info, g_reward, g_pi0_reward, action_pi0, next_x_pi0
    

class EpisodeLogger:
    def __init__(self, folder):
        self.folder = folder
        self.data = {}

    def log(self, key, array):
        """
        Log a numpy array for a given key.

        Parameters:
        - key (str): Key to identify the array.
        - array (numpy.ndarray): Numpy array to be logged.
        """
        if key in self.data:
            self.data[key] = np.vstack((self.data[key], array))
        else:
            self.data[key] = array.reshape(-1, 1)

    def save_to_file(self, ep_num):
        """
        Save the logged data to a CSV file.
        """
        file_path = os.path.join(self.folder, f'episode_{ep_num}.csv')

        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            header = ['Index'] + list(self.data.keys())
            writer.writerow(header)

            # Write data
            max_length = max(len(arr) for arr in self.data.values())
            for i in range(max_length):
                row = [i]
                for key in self.data.keys():
                    if key in self.data and i < len(self.data[key]):
                        row.append(self.data[key][i][0])  # Assuming 1D arrays
                    else:
                        row.append(None)
                writer.writerow(row)

    def clear_data(self):
        """
        Clear all the logged data.
        """
        self.data = {}


class AvgQvals:
    def __init__(self):
        self.num_g0 = 0
        self.num_g1 = 0
        self.sum_g0 = 0.0
        self.sum_g1 = 0.0

    def update(self, g0, g1):
        self.num_g0 += len(g0)
        self.num_g1 += len(g1)
        self.sum_g0 += g0.sum()
        self.sum_g1 += g1.sum()

    def avg_g0(self):
        return self.sum_g0 / self.num_g0
    
    def avg_g1(self):
        return self.sum_g1 / self.num_g1
    
class RollingAvg:
    '''
        class used for tracking rolling avg for logging
    '''
    def __init__(self, n):
        self.n = n
        self.policy_constraint = []
        self.soft_ie = []
        self.soft_se = []
        self.soft_de = []
        self.c_pi_theta = []
        self.lambda_loss = []
        self.dpe_loss = []

    def update(self, policy_constraint, soft_ie, soft_se, soft_de, c_pi_theta, lambda_loss, dpe_loss):
        self.policy_constraint.append(policy_constraint)
        self.soft_ie.append(soft_ie)
        self.soft_se.append(soft_se)
        self.soft_de.append(soft_de)
        self.c_pi_theta.append(c_pi_theta)
        self.lambda_loss.append(lambda_loss)
        self.dpe_loss.append(dpe_loss)

        if len(self.policy_constraint) > self.n:
            self.policy_constraint.pop(0)
            self.soft_ie.pop(0)
            self.soft_se.pop(0)
            self.soft_de.pop(0)
            self.c_pi_theta.pop(0)
            self.lambda_loss.pop(0)
            self.dpe_loss.pop(0)

    def avg_policy_constraint(self):
        return np.mean(self.policy_constraint)
    
    def avg_soft_ie(self):
        return np.mean(self.soft_ie)
    
    def avg_soft_se(self):
        return np.mean(self.soft_se)
    
    def avg_soft_de(self):
        return np.mean(self.soft_de)
    
    def avg_c_pi_theta(self):
        return np.mean(self.c_pi_theta)
    
    def avg_lambda_loss(self):
        return np.mean(self.lambda_loss)
    
    def avg_dpe_loss(self):
        return np.mean(self.dpe_loss)
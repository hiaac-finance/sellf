from typing import cast, Optional, List, Union, Dict, Any
from dataclasses import dataclass, field
from omegaconf import OmegaConf, MISSING
import argparse
import os
import zipfile
from datetime import datetime
import random
import shutil
from pathlib import Path

import pandas as pd
import numpy as np
import pickle
import torch
import tqdm


from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from environments.lending import DelayedImpactEnv
from environments.lending_params import DelayedImpactParams, two_group_credit_clusters
from environments.rewards import LendingReward
from agents.ppo_pred.ppo_pred import PPOPred
from agents.ppo_pred.buffer import RolloutBuffer
from agents.ppo_pred.wrapper_env import PPOPredEnvWrapper
from graphing.plot_all import complete_plot_reward_mu
from omegaconf import OmegaConf


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device: ', device)
torch.cuda.empty_cache()

import logging

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s")

logger = logging.getLogger(__name__)



@dataclass
class GeneralParams:
    seed: Optional[int] = None
    algorithm: str = MISSING
    exp_dir: str = MISSING
    n_seeds: int = 1

@dataclass
class EnvironmentParams:
    partial_observation: bool = MISSING
    cluster_probabilities: List[List[float]] = MISSING
    group_likelihoods: List[float] = MISSING
    success_probabilities: List[List[float]] = MISSING
    credit_drift_probs: List[List[float]] = MISSING
    bank_starting_cash: float = MISSING
    interest_rate: float = MISSING
    cluster_shift_increment: float = MISSING
    ep_timesteps: int = MISSING
    omega: float = MISSING

@dataclass
class AlgorithmParams:
    policy: str = MISSING
    learning_rate: float = MISSING
    n_steps: int = MISSING
    batch_size: int = MISSING
    n_epochs: int = MISSING
    train_timesteps: int = MISSING
    eval_timesteps: int = MISSING
    disp_coef: float = 0.5
    pred_coef: float = 0.2

@dataclass
class PolicyParams:
    activation_fn: str = MISSING
    net_arch: List[Any] = MISSING

@dataclass
class Config:
    general: GeneralParams = field(default_factory=GeneralParams)
    environment: EnvironmentParams = field(default_factory=EnvironmentParams)
    algorithm: AlgorithmParams = field(default_factory= AlgorithmParams)
    policy: PolicyParams = field(default_factory=PolicyParams)

template = Config()

def validate_config(parameters):
    if isinstance(parameters, dict):
        params = OmegaConf.create(parameters)
    
    elif isinstance(parameters, (str, Path)):
        params = OmegaConf.load(parameters)
    
    else:
        raise ValueError("Invalid parameters type")

    validated_config = OmegaConf.merge(template, params)
    validated_config = cast(Config, validated_config)

    return validated_config


def save_code(save_dir):
    print('Saving code...')
    code_dir = './'
    date_time_str = datetime.now().strftime("%b%d_%Y_at_%H_%M_%S")
    ignored_directories = ['.git', '__pycache__', 'evaluations_old', 'computation_graphs']
    ignore_contains = ['results']
    ignore_extensions = ['.ipynb']

    zip_file = os.path.join(save_dir, f'code_{date_time_str}.zip')
    with zipfile.ZipFile(zip_file, 'w') as zipf:
        for root, dirs, files in os.walk(code_dir):
            # Remove hidden directories and directories containing the specific string
            dirs[:] = [d for d in dirs if not (d.startswith('.') or any(substring in d for substring in ignore_contains) or d in ignored_directories)]
            for file in files:
                file_path = os.path.join(root, file)
                if not file.endswith(tuple(ignore_extensions)):
                    zipf.write(file_path, os.path.relpath(file_path, code_dir))

    print('Code saved to: ', zip_file)


def train_multi(
    config,
    env_params,
):
    print('env_params: ', env_params)

    model = None
    
    save_dir = os.path.join(config.general.exp_dir, config.general.algorithm, "models")
    shutil.rmtree(os.path.join(config.general.exp_dir, config.general.algorithm), ignore_errors=True)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    save_code(save_dir)

    with open(os.path.join(save_dir, 'config.txt'), 'w') as f:
        f.write('----------------------------------------------------\n')
        f.write('----------------------------------------------------\n')
        date_time_str = datetime.now().strftime("%b %d, %Y at %H:%M:%S")
        f.write(date_time_str + '\n\n')
        keys = [key for key in dir(config) if not key.startswith('__')]
        for key in keys:
            f.write(f'{key}: {getattr(config,key)}\n')

        f.write(f'\n')

    seeds = [random.randint(0, 10000) for i in range(config.general.n_seeds)]

    for i in range(config.general.n_seeds):
        env = DelayedImpactEnv(env_params)
        seed = seeds[i]
        env.seed(seed)
        env = PPOPredEnvWrapper(env=env, reward_fn=LendingReward, omega=config.environment.omega, dist_test=True)
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        
        with open(os.path.join(save_dir, 'config.txt'), 'a') as f:
            f.write(f'SEED{i}: {seed}\n')
        
        save_dir = os.path.join(save_dir, f'seed_{seed}')
        activation_fn = getattr(torch.nn, config.policy.activation_fn)
        model = PPOPred(
            env = env,
            device = device,
            policy_kwargs={
                "activation_fn" : activation_fn,
                "net_arch" : [256, 256, dict(vf=[256, 128], pi=[256, 128])],#params.policy.net_arch.to_container()
            },
            **config.algorithm
        )

        checkpoint_callback = CheckpointCallback(save_freq=config.algorithm.train_timesteps, save_path=save_dir,
                                                name_prefix='rl_model')

        model.set_logger(configure(folder=save_dir, format_strings=['log', 'csv']))
        
        model.learn(total_timesteps=config.algorithm.train_timesteps) #, callback=checkpoint_callback)
        model.save(os.path.join(save_dir, 'final_model'))

        # Once we finish learning, plot the returns over time and save into the experiments directory
        #try:
        #    plot_rets(save_dir)
        #except:
        #    plt.close()
        #    print(f'Could not plot returns for {save_dir}')

        # column_list = [ 'train/cumulative_gx_g0', 'train/cumulative_gx_g1',
        #             'rollout/ep_rew_mean', 'train/policy_constraint_loss', 'train/lambda_loss',
        #     'train/policy_gradient_loss', 'train/value_loss',
        #     'train/static_kl', 'train/cumulative_reward', 'train/soft_de', 'train/soft_ie', 'train/soft_se', 'train/c_pi_theta',
        #     'train/rolling_soft_de', 'train/rolling_soft_ie', 'train/rolling_soft_se', 'train/rolling_c_pi_theta']
        # try:
        #     plot_progress_data(save_dir, column_list)
        # except:
        #     plt.close()
        #     print(f'Could not plot progress data for {save_dir}')

    #try:
    #    plot_multi_seed_progress_data(SAVE_DIR, seeds, column_list)
    #except:
    #    plt.close()
    #    print(f'Could not plot multi_seed_progress_data for {SAVE_DIR}')
    return seeds
    


def evaluate(env, agent, num_eps, seeds, eval_path, config):
    eval_data = []
    for ep in range(num_eps):
        random.seed(seeds[ep])
        np.random.seed(seeds[ep])
        torch.manual_seed(seeds[ep])

        env.seed(seeds[ep])

        obs = env.reset()
        bank_starting_cash = env.state.bank_cash
        
        done = False
        #loans_ot_by_cscore = np.zeros((NUM_GROUPS, num_cscores))
        #cscore_seen_over_time = np.zeros((NUM_GROUPS, num_cscores))
        #dummy_buff = DummyEvalBuffer(num_timesteps, obs.shape, 2, 1, 100, config_params.DELAYED_IMPACT_CLUSTER_PROBS, config_params.QUAL_CHANGE)
        for t in tqdm.trange(config.algorithm.eval_timesteps):

            action = agent.predict(obs)[0]
            #if algorithm == 'cpo':
            #    action = int(agent(torch.FloatTensor(obs).squeeze()).sample().item())
            #else:
            #    if isinstance(agent, PPO):
            #        action = agent.predict(obs)[0]
            #    else:
            #        action = agent.act(obs, done)

            # Logging

            next_obs, rew, done, info = env.step(action)
            bank_cash = env.state.bank_cash
            obs = next_obs

            eval_data.append({
                "ep": ep,
                "t" : t,
                "bank_cash": bank_cash - bank_starting_cash,
                "mu0" : env.mu[0],
                "mu1" : env.mu[1],
                "mu0_obs" : env.mu_obs[0],
                "mu1_obs" : env.mu_obs[1],
                "delta" : env.delta,
                "delta_obs" : env.delta_obs,
            })
            if done:
                break

    # ------------------- new part -------------------------
    if not os.path.isdir(eval_path):
        os.makedirs(eval_path, exist_ok=True)

    with open(os.path.join(eval_path, 'eval_data.pkl'), 'wb') as f:
        pickle.dump(eval_data, f)
    # ------------------------------------------------------

    return eval_data


def main(config, arg_train=False, arg_eval=False, is_outside_func_call=False):
    if is_outside_func_call:
        print('Is outside function call...')
        random.seed(config.general.seed)
        np.random.seed(config.general.seed)
        torch.manual_seed(config.general.seed)


    env_params = DelayedImpactParams(
        applicant_distribution=two_group_credit_clusters(
            cluster_probabilities=config.environment.cluster_probabilities,
            group_likelihoods=config.environment.group_likelihoods,
            success_probabilities=config.environment.success_probabilities,
            credit_drift_probs=config.environment.credit_drift_probs,),
        bank_starting_cash=config.environment.bank_starting_cash,
        interest_rate=config.environment.interest_rate,
        cluster_shift_increment=config.environment.cluster_shift_increment,
    )
    env = DelayedImpactEnv(env_params)
    env.seed(config.general.seed)

    t_seeds = train_multi(config, env_params)

    # Initialize eval directory to store eval information
    eval_dir = os.path.join(config.general.exp_dir, config.general.algorithm, "evaluation")
    shutil.rmtree(eval_dir, ignore_errors=True)
    Path(eval_dir).mkdir(parents=True, exist_ok=True)

    # Get random seeds
    eval_eps = 5 # 10
    seeds = [random.randint(0, 10000) for _ in range(eval_eps)]

    with open(os.path.join(eval_dir, 'seeds.txt'), 'w') as f:
        f.write(str(seeds)+"\n")
        f.write(str(config))

    eval_paths = []



    if config.general.n_seeds != 1:
        weights_step = model_path.split('/')[-1]
        base_path = config.SAVE_DIR
        seed_dirs = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

        for seed_dir in seed_dirs:
            model_path = os.path.join(base_path, seed_dir, weights_step)
            env = DelayedImpactEnv(env_params)
            agent = PPOPred.load(model_path, device=device, verbose=1)

            # I think PPO.load converts keys to strings b/c json stuff, want keys to be ints
            new_ben_dict = {}
            for k, v in agent.benefit_deltas_dict.items(): 
                for x_k, x_v in v.items():
                    if int(k) not in new_ben_dict:
                        new_ben_dict[int(k)] = {}
                    new_ben_dict[int(k)][int(x_k)] = x_v
            agent.benefit_deltas_dict = new_ben_dict 

            m_name = os.path.join(name, seed_dir)

            eval_data = evaluate(env=PPOEnvWrapper(env=env, reward_fn=LendingReward, config_params=conf, is_eval=True),
                    agent=agent,
                    num_eps=eval_eps,
                    name=m_name,
                    seeds=seeds,
                    eval_path=os.path.join(args.eval_path, m_name),
                    config_params=conf,
            )
            eval_paths.append(os.path.join(args.eval_path, m_name))
            
    else:
        base_path = os.path.join(config.general.exp_dir, config.general.algorithm, "models")
        seed_dirs = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
        seed_dir = seed_dirs[0]
        model_path = os.path.join(base_path, seed_dir, "final_model")
        env = DelayedImpactEnv(env_params)
        agent = PPOPred.load(model_path, verbose=1)
        env = PPOPredEnvWrapper(env, reward_fn = LendingReward, omega=config.environment.omega, dist_test=True, ep_timesteps = config.algorithm.eval_timesteps)

        eval_dir = os.path.join(config.general.exp_dir, config.general.algorithm, "evaluation")
        evaluate(
                env=env,
                agent=agent,
                num_eps=eval_eps,
                seeds=seeds,
                eval_path=eval_dir,
                config=config,
            )
        eval_paths.append(eval_dir)


    for path in eval_paths:
        with open(os.path.join(path, 'eval_data.pkl'), 'rb') as f:
            eval_data = pickle.load(f)
        eval_data = pd.DataFrame(eval_data)
        complete_plot_reward_mu(eval_data, path)

    

def validate_config(parameters):
    if isinstance(parameters, dict):
        params = OmegaConf.create(parameters)
    
    elif isinstance(parameters, (str, Path)):
        params = OmegaConf.load(parameters)
    
    else:
        raise ValueError("Invalid parameters type")

    #validated_config = OmegaConf.merge(template, params)
    #validated_config = cast(Config, validated_config)

    return params

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', nargs='?',type=str, help='Path to the config file')
    parser.add_argument('--sweeps', '-s', action='store_true', help='Run sweeps')
    #parser.add_argument('--train', action='store_true', default=False)
    #parser.add_argument('--eval', action='store_true', default=False)
    args = parser.parse_args()

    if args.config_path is None:
        logger.info("No config file provided, using default config")
        config = "config_files/ppopred.yaml"
    else:
        config = args.config_path


    #random.seed(config.seed)
    #np.random.seed(config.seed)
    #torch.manual_seed(config.seed)
    # torch.cuda.manual_seed(SEED)
    
    config = validate_config(config)
    main(config)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import functools
import os
import random
import shutil
from pathlib import Path
import time
from omegaconf import OmegaConf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pandas as pd
import torch
import tqdm
import pickle as pkl
from absl import flags
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from yaml import full_load

import sys; sys.path.append('..')

from lending_experiment.environments import params, rewards
from lending_experiment.environments.lending import DelayedImpactEnv, EnemEnv
from lending_experiment.environments.lending_params import DelayedImpactParams, two_group_credit_clusters
from lending_experiment.environments.rewards import LendingReward
from lending_experiment.agents.ppo.ppo_wrapper_env import PPOEnvWrapper
from lending_experiment.agents.ppo.sb3.ppo import PPO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device: ', device)
torch.cuda.empty_cache()

def get_env(env_name: str):
    if env_name == "yu2022":
        env_params = DelayedImpactParams(
            applicant_distribution=two_group_credit_clusters(
                cluster_probabilities=[
                    [0.0, 0.1, 0.1, 0.2, 0.3, 0.3, 0.0],
                    [0.1, 0.1, 0.2, 0.3, 0.3, 0.0, 0.0],
                ],
                group_likelihoods=[0.5, 0.5],
                success_probabilities=[
                    [0.1, 0.2, 0.45, 0.6, 0.65, 0.7, 0.7], 
                    [0.1, 0.2, 0.45, 0.6, 0.65, 0.7, 0.7]
                ]
            ),
            bank_starting_cash=10_000,
            interest_rate=1,
            cluster_shift_increment=0.01,
        )
        env = DelayedImpactEnv(env_params)
    elif env_name == "yu2022_hard":
        env_params = DelayedImpactParams(
            applicant_distribution=two_group_credit_clusters(
                cluster_probabilities=[
                    [0.0, 0.1, 0.1, 0.2, 0.3, 0.3, 0.0],
                    [0.1, 0.1, 0.2, 0.3, 0.3, 0.0, 0.0],
                ],
                group_likelihoods=[0.7, 0.3],
                success_probabilities=[
                    [0.1, 0.2, 0.45, 0.6, 0.65, 0.7, 0.7], 
                    [0.1, 0.1, 0.25, 0.4, 0.75, 0.8, 0.8]
                ]
            ),
            bank_starting_cash=10_000,
            interest_rate=1,
            cluster_shift_increment=0.01,
        )
        env = DelayedImpactEnv(env_params)
    elif env_name == "setting1_hard":
        env_params = DelayedImpactParams(
            applicant_distribution=two_group_credit_clusters(
                cluster_probabilities=[
                    (0.0, 0.0, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15, 0.0, 0.0),
	                (0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15, 0.0, 0.0, 0.0, 0.0),  
                ],
                group_likelihoods=[0.5, 0.5],
                success_probabilities=[
                    (0.773, 0.804, 0.833, 0.857, 0.879, 0.898, 0.914, 0.928, 0.939, 0.949, 0.958, 0.965, 0.970, 0.975),
                    (0.673, 0.704, 0.733, 0.757, 0.779, 0.798, 0.714, 0.728, 0.939, 0.949, 0.958, 0.965, 0.970, 0.975),
                ]
            ),
            bank_starting_cash=10_000,
            interest_rate=0.1,
            cluster_shift_increment=0.01,
        )
        env = DelayedImpactEnv(env_params)

    elif env_name == "fico":
        with open("data/fico.pkl", "rb") as f:
            data = pkl.load(f)

        env_params = DelayedImpactParams(
            applicant_distribution=two_group_credit_clusters(
                cluster_probabilities=data["cluster_probabilities"],
                group_likelihoods=data["group_likelihoods"],
                success_probabilities=data["success_probabilities"]
            ),
            bank_starting_cash=10_000,
            interest_rate=0.25,
            cluster_shift_increment=0.01,
        )
        env = DelayedImpactEnv(env_params)

    elif env_name == "enem":
        with open("data/enem.pkl", "rb") as f:
            data = pkl.load(f)

        env_params = DelayedImpactParams(
            applicant_distribution=two_group_credit_clusters(
                cluster_probabilities=data["cluster_probabilities"],
                group_likelihoods=data["group_likelihoods"],
                success_probabilities=data["success_probabilities"]
            ),
            bank_starting_cash=10_000,
            interest_rate=10,
            cluster_shift_increment=0.01,
        )
        env = EnemEnv(env_params)
    return env


def train(train_timesteps, env, config):

    exp_dir = os.path.join(config.general.exp_dir, config.general.env_name)
    save_dir = os.path.join(exp_dir, config.general.algorithm, "models")

    print('env_params: ', env.state.params)

    env = PPOEnvWrapper(env=env, reward_fn=LendingReward, mu_type=config.environment.mu_type)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])


    model = PPO(
        "MlpPolicy", 
        env,
        policy_kwargs={
            "use_predictor": config.policy.use_predictor,
            "activation_fn": torch.nn.ReLU,
            "net_arch": [256, 256, dict(vf=[256, 128], pi=[256, 128])],
        },
        verbose=0,
        device=device,
        **config.algorithm
    )

    shutil.rmtree(save_dir, ignore_errors=True)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    checkpoint_callback = CheckpointCallback(save_freq=10_000, save_path=save_dir,
                                             name_prefix='rl_model')

    model.set_logger(configure(folder=save_dir))
    model.learn(total_timesteps=train_timesteps, callback=checkpoint_callback)
    model.save(save_dir + '/final_model')

    # Once we finish learning, plot the returns over time and save into the experiments directory
    #plot_rets(EXP_DIR)

def evaluate(env, agent, num_eps, num_timesteps, name, seeds, eval_path):
    print()
    print(f"Evaluating {name}")
    Path(f'{eval_path}/{name}/').mkdir(parents=True, exist_ok=True)
    eval_data = []

    reward_fn = LendingReward()

    for ep in range(num_eps):
        random.seed(seeds[ep])
        np.random.seed(seeds[ep])
        torch.manual_seed(seeds[ep])

        obs = env.reset()
        done = False
        print(f'Episode {ep}:')
        for t in tqdm.trange(num_timesteps):
            will_default = env.state.will_default


            action = None
            if isinstance(agent, PPO):
                action = agent.predict(obs)[0]
            else:
                action = agent.act(obs, done)
            obs = np.array(obs).reshape(1, -1)
            obs = torch.tensor(obs, dtype=torch.float32).to(device)
            pred = agent.policy.predict_label(obs).item()

            # Logging
            group_id = np.argmax(env.state.group)
            # Add to loans if the agent wants to loan
            label = 1 - env.state.will_default

            env.pred = pred

            old_bank_cash = env.state.bank_cash

            obs, _, done, _ = env.step(action)

            bank_cash = env.state.bank_cash

            r = reward_fn(old_bank_cash=old_bank_cash,
                          bank_cash=bank_cash,
                          tpr=np.zeros(2,),#eval_data['tot_tpr'][ep],
                          zeta0=1,
                          zeta1=0)
            
            eval_data.append({
                "ep" : ep,
                "t" : t,
                "group_id" : group_id,
                "action" : action,
                "label" : label,
                "pred" : pred,
                "correct" : int(label == pred),
                "bank_cash" : bank_cash,
                "mu0" : env.mu[0],
                "mu1" : env.mu[1],
                "delta" : env.delta,
                "delta_real" : env.delta_real,
            })
        
            if done:
                break

                
    predictions = []
    # calculate the output of the predictor net
    for g in range(env.state.params.num_groups):
        for x in range(env.env.observation_space['applicant_features'].shape[0]):
            # transform g to one hot encode
            g_ = np.zeros(2)
            g_[g] = 1
            x_ = np.zeros(env.env.observation_space['applicant_features'].shape[0])
            x_[x] = 1
            obs = np.concatenate([x_, g_])
            obs = np.expand_dims(obs, axis=0)
            obs = torch.tensor(obs, dtype=torch.float32).to(device)
            pred = agent.policy.prob_label(obs).cpu().detach().numpy()
            predictions.append({
                "g": g,
                "x": x,
                "pred" : pred.item()
            })

    Path(f'{eval_path}').mkdir(parents=True, exist_ok=True)
    eval_data = pd.DataFrame(eval_data)
    eval_data.to_csv(f'{eval_path}/eval_data.csv', index=False)

    predictions = pd.DataFrame(predictions)
    predictions.to_csv(f'{eval_path}/predictions.csv', index=False)

    return eval_data


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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", nargs="?", type=str, help="Path to the config file")
    args = parser.parse_args()

    if args.config_path is None:
        config = "config_files/yu2022/ppo.yaml"
    else:
        config = args.config_path
    
    config = validate_config(config)

    start = time.time()
    env = get_env(config.general.env_name)
    train(train_timesteps=config.algorithm.train_timesteps, env=env, config=config)
    #    plot_rets(exp_path=EXP_DIR, save_png=True)

    #if args.show_train_progress:
    #    plot_rets(exp_path=EXP_DIR, save_png=False)

    #if args.display_eval_path is not None:
    #    display_eval_results(eval_dir=args.display_eval_path)

    #print(args.eval_path)

    eval_path = os.path.join(config.general.exp_dir, config.general.env_name, config.general.algorithm, "eval")

    # Initialize eval directory to store eval information
    shutil.rmtree(eval_path, ignore_errors=True)
    Path(eval_path).mkdir(parents=True, exist_ok=True)

    # Get random seeds
    eval_eps = 5
    eval_timesteps = 10_000
    seeds = [random.randint(0, 10000) for _ in range(eval_eps)]

    with open(eval_path + '/seeds.txt', 'w') as f:
        f.write(str(seeds))

    model_path = os.path.join(
        config.general.exp_dir,
        config.general.env_name,
        config.general.algorithm,
        "models",
        "final_model.zip"
    )
    env = get_env(config.general.env_name)
    name = config.general.algorithm
    agent = PPO.load(model_path, verbose=1)
    evaluate(
        env=PPOEnvWrapper(env=env, reward_fn=LendingReward, ep_timesteps=eval_timesteps, mu_type=config.environment.mu_type),
        agent=agent,
        num_eps=eval_eps,
        num_timesteps=eval_timesteps,
        name=name,
        seeds=seeds,
        eval_path=eval_path
    )

    end = time.time()
    with open("log.txt", "a") as f:
        f.write(f"Environment: {config.general.env_name}\n")
        f.write(f"Algorithm: {config.general.algorithm}\n")
        f.write(f"Train Timesteps: {config.algorithm.train_timesteps}\n")
        f.write(f"Took: {(end - start) / 60} minutes\n")
        f.write("\n")


if __name__ == '__main__':
    main()
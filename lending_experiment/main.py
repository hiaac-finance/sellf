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
#from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from yaml import full_load

import sys; sys.path.append('..')

from lending_experiment.agents.ppo.ppo_wrapper_env import PPOEnvWrapper, Monitor
from lending_experiment.agents.ppo.sb3.ppo import PPO

from lending_experiment.environments.resampling import ResamplingEnv, LendingEnv, Params as ResamplingParams

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device: ', device)
torch.cuda.empty_cache()

def get_env(env_name: str, utility_method: str, delta_method: str) -> ResamplingEnv:
    params = ResamplingParams()
    env = LendingEnv(params, utility_method=utility_method, delta_method=delta_method)
    return env


def train(train_timesteps, env, config):

    exp_dir = os.path.join(config.general.exp_dir, config.general.env_name)
    save_dir = os.path.join(exp_dir, config.general.algorithm, "models")

    print('env_params: ', env.state.params)

    env = PPOEnvWrapper(
        env=env, 
        ep_timesteps=config.environment.ep_timesteps,
    )
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])


    model = PPO(
        "MlpPolicy", 
        env,
        policy_kwargs={
            "use_predictor": config.policy.use_predictor,
            "activation_fn": torch.nn.ReLU,
            "net_arch": [32, dict(vf=[32], pi=[32])],
        },
        verbose=0,
        omega=config.environment.omega,
        device=device,
        **config.algorithm
    )
    env.env_method("set_agent", model)

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


    for ep in range(num_eps):
        random.seed(seeds[ep])
        np.random.seed(seeds[ep])
        torch.manual_seed(seeds[ep])

        obs = env.reset()
        done = False
        print(f'Episode {ep}:')

        # Make predictions for everyone
        action_list = []
        pred_list = []
        for idx in range(env.get_attr("num_applicants")[0]):
            obs = env.env_method("get_applicant_obs", idx)[0]
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            obs_tensor = obs_tensor.unsqueeze(0)
            action, _, _ = agent(obs_tensor)
            action = action.cpu().numpy()
            pred = agent.predict_label(obs_tensor).item()
            action_list.append(action)
            pred_list.append(pred)
        
        env.env_method("set_action_pred", action_list, pred_list)


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

   
    Path(f'{eval_path}').mkdir(parents=True, exist_ok=True)
    eval_data = pd.DataFrame(eval_data)
    eval_data.to_csv(f'{eval_path}/eval_data.csv', index=False)

    # if env.env.observation_space["applicant_features"].shape[0] > 10:
    #     return eval_data

    # predictions = []
    # # calculate the output of the predictor net
    # for g in range(env.state.params.num_groups):
    #     for x in range(env.env.observation_space['applicant_features'].shape[0]):
    #         # transform g to one hot encode
    #         g_ = np.zeros(2)
    #         g_[g] = 1
    #         x_ = np.zeros(env.env.observation_space['applicant_features'].shape[0])
    #         x_[x] = 1
    #         obs = np.concatenate([x_, g_])
    #         obs = np.expand_dims(obs, axis=0)
    #         obs = torch.tensor(obs, dtype=torch.float32).to(device)
    #         pred = agent.policy.prob_label(obs).cpu().detach().numpy()
    #         #action = agent.policy.prob_loan(obs)[0]
    #         predictions.append({
    #             "g": g,
    #             "x": x,
    #             "pred" : pred.item(),
    #         #    "action" : action.item(),
    #         })

    # predictions = pd.DataFrame(predictions)
    # predictions.to_csv(f'{eval_path}/predictions.csv', index=False)

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
    with open("log.txt", "a") as f:
        f.write("\n-------------------------------------------------------------\n")
        f.write(f"Start time: {time.strftime('%H:%M:%S', time.localtime(start))}\n")
        f.write(f"Environment: {config.general.env_name}\n")
        f.write(f"Algorithm: {config.general.algorithm}\n")
        f.write(f"Train Timesteps: {config.algorithm.train_timesteps}\n")
        
    env = get_env(config.general.env_name, config.environment.mu_type, config.environment.obs_type)
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
    env = get_env(config.general.env_name, config.environment.mu_type, config.environment.obs_type)
    name = config.general.algorithm
    agent = PPO.load(model_path, verbose=1)
    env = PPOEnvWrapper(
        env=env, 
        ep_timesteps=config.environment.ep_timesteps,
    )
    env.set_agent(agent)
    evaluate(
        env=env,
        agent=agent,
        num_eps=eval_eps,
        num_timesteps=config.environment.ep_timesteps,
        name=name,
        seeds=seeds,
        eval_path=eval_path
    )

    end = time.time()
    with open("log.txt", "a") as f:
        f.write(f"End time: {time.strftime('%H:%M:%S', time.localtime(end))}\n")
        f.write(f"Took: {(end - start) / 60} minutes\n")
        f.write("\n")


if __name__ == '__main__':
    main()
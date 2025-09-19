import os
import random
import shutil
from pathlib import Path
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

import sys

sys.path.append("..")

from lending_experiment.agents.ppo_wrapper_env import PPOEnvWrapper
from lending_experiment.agents.ppo import PPO
from lending_experiment.agents.pocar import POCAR
from lending_experiment.agents.rrm import RRM
from lending_experiment.agents.sellf import SELLF
from lending_experiment.agents.serew import SEREW
from lending_experiment.agents.simple_agent import SimpleAgent

from lending_experiment.environments.resampling import (
    ResamplingEnv,
    LendingEnv,
    EnemEnv,
)
import argparse
from omegaconf import OmegaConf


import torch.multiprocessing as mp

EXP_DIR = "./experiments"


def get_env(env_name: str, utility_method: str, algorithm: str) -> ResamplingEnv:
    if algorithm in ["pocar_full", "ppo", "rrm", "simple_agent"]:
        delta_method = "full"
    elif algorithm == "sellf":
        delta_method = "imputation"
    elif algorithm == "serew":
        delta_method = "imputation"
    elif algorithm == "sellf_hard":
        delta_method = "imputation_hard"
    else:
        delta_method = "accepted"
    if "fico" in env_name or "setting" in env_name:
        env = LendingEnv(
            utility_method=utility_method,
            delta_method=delta_method,
            distributions=env_name,
            n_applicants=4_000,
            seed=0,
        )
    elif env_name == "enem":
        env = EnemEnv(utility_method=utility_method, delta_method=delta_method, seed=0)
    if algorithm == "pocar_full":
        env.next_disp = True
    return env


def get_alg(env, config, device):
    if config["algorithm"] == "ppo":
        model = PPO(
            env=env,
            policy_kwargs={
                "use_predictor": config["use_predictor"],
            },
            device=device,
            **config["algorithm_params"],
        )
    elif config["algorithm"] == "simple_agent":
        model = SimpleAgent(
            env=env,
            policy_kwargs={
                "use_predictor": config["use_predictor"],
            },
            device=device,
            **config["algorithm_params"],
        )
    elif config["algorithm"] == "pocar" or config["algorithm"] == "pocar_full":
        model = POCAR(
            env=env,
            policy_kwargs={
                "use_predictor": config["use_predictor"],
            },
            omega=config["omega"],
            device=device,
            **config["algorithm_params"],
        )
    elif config["algorithm"] == "rrm":
        model = RRM(
            env=env,
            policy_kwargs={
                "use_predictor": config["use_predictor"],
            },
            omega=config["omega"],
            device=device,
            **config["algorithm_params"],
        )
    elif config["algorithm"].find("sellf") != -1:
        model = SELLF(
            env=env,
            policy_kwargs={
                "use_predictor": config["use_predictor"],
            },
            omega=config["omega"],
            device=device,
            **config["algorithm_params"],
        )
    elif config["algorithm"] == "serew":
        model = SEREW(
            env=env,
            policy_kwargs={
                "use_predictor": True,
            },
            omega=config["omega"],
            device=device,
            **config["algorithm_params"],
        )

    return model


def train(train_timesteps, env, save_dir, config, device):

    env = PPOEnvWrapper(env=env)
    env = Monitor(env)

    model = get_alg(env, config, device)
    model.set_random_seed(0)
    env.set_agent(model)

    shutil.rmtree(save_dir, ignore_errors=True)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    model.set_logger(configure(folder=save_dir, format_strings=["csv"]))
    model.learn(total_timesteps=train_timesteps)
    model.save(save_dir + "/final_model")


def plot_learning(env_name, mu_type, alg_name):
    df_log = pd.read_csv(
        f"experiments/{env_name}/{mu_type}/{alg_name}/models/progress.csv"
    )

    # first, count the columns of the dataframe
    columns = sorted(df_log.columns.tolist())

    # separated columns that end with "g0" or "g1"
    columns_g = [col for col in columns if col.endswith("g0") or col.endswith("g1")]
    columns_g = sorted(columns_g)

    columns = [col for col in columns if col not in columns_g]

    # create pairs of sequential columns
    columns_g = [columns_g[i : i + 2] for i in range(0, len(columns_g), 2)]

    columns = columns + columns_g

    n_rows = len(columns) // 5 + (len(columns) % 5 > 0)
    n_cols = 5
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(13, int(3 * n_rows)))

    axs = axs.flatten()

    # add 0 if column is not present
    for col in columns:
        if isinstance(col, list):
            for sub_col in col:
                if sub_col not in df_log.columns:
                    df_log[sub_col] = 0
        else:
            if col not in df_log.columns:
                df_log[col] = 0

    def rolling_mean(series, window=10):
        return series.rolling(window=window, min_periods=1).mean()

    for i, col in enumerate(columns):
        if isinstance(col, list):
            for j, sub_col in enumerate(col):
                data = rolling_mean(df_log[sub_col])
                axs[i].plot(data, label=sub_col)

            title = col[0].replace("0", "_i").replace("1", "_i")
            axs[i].set_title(title)
        else:
            data = rolling_mean(df_log[col])
            axs[i].plot(data, label=col)
            axs[i].set_title(col)

        # if ylim contains 0, draw a line at 0
        ylim = axs[i].get_ylim()
        if ylim[0] * ylim[1] < 0:
            axs[i].axhline(0, color="black", linestyle="--")

        if col == "train/pred_lr":
            axs[i].set_yscale("log")

        axs[i].set_xlabel("Training Steps")
        axs[i].set_ylabel("Value")
        # axs[i].legend()
    plt.tight_layout()
    plt.savefig(f"experiments/{env_name}/{mu_type}/{alg_name}/models/learning.png")


def evaluate(env, agent, seeds, eval_dir, device):
    eval_data = []
    num_eps = len(seeds)
    for ep in range(num_eps):
        random.seed(seeds[ep])
        np.random.seed(seeds[ep])
        torch.manual_seed(seeds[ep])
        env.seed(seeds[ep])

        env.update_models()
        obs = env.reset()
        done = False
        t = 0
        while not done:
            obs = np.array(obs).reshape(1, -1)
            obs = torch.tensor(obs, dtype=torch.float32).to(device)
            action = agent.get_action(obs)
            action = action.item()
            pred = agent.get_label(obs).item()

            # Logging
            group_id = env.data["group"][env.idx]
            # Add to loans if the agent wants to loan
            label = env.data["label"][env.idx]
            obs, _, done, _ = env.step(action)
            resource = env.resource
            eval_data.append(
                {
                    "ep": ep,
                    "t": t,
                    "group_id": group_id,
                    "action": action,
                    "label": label,
                    "pred": pred,
                    "correct": int(label == pred),
                    "resource": resource,
                    "delta": env.delta,
                    "delta_obs": env.delta_obs,
                }
            )
            t += 1
            if done:
                break

    eval_data = pd.DataFrame(eval_data)
    eval_data.to_csv(f"{eval_dir}/eval_data.csv", index=False)

    return eval_data


def main(config):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    torch.set_num_threads(1)

    device = torch.device("cpu")

    start = time.time()
    with open("log.txt", "a") as f:
        f.write("\n-------------------------------------------------------------\n")
        f.write(f"Start time: {time.strftime('%H:%M:%S', time.localtime(start))}\n")
        f.write(f"Config: {config}\n")
        f.write("\n-------------------------------------------------------------\n")

    exp_dir = (
        f"./experiments/{config['env_name']}/{config['mu_type']}/{config['exp_name']}"
    )
    Path(exp_dir).mkdir(parents=True, exist_ok=True)
    save_dir = f"{exp_dir}/models"
    eval_dir = f"{exp_dir}/eval"
    config["use_predictor"] = config["algorithm"].find("sellf") != -1
    env = get_env(config["env_name"], config["mu_type"], config["algorithm"])

    # set seeds
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    train(
        train_timesteps=config["train_timesteps"],
        env=env,
        save_dir=save_dir,
        config=config,
        device=device,
    )

    plot_learning(config["env_name"], config["mu_type"], config["exp_name"])
    
    # Initialize eval directory to store eval information
    shutil.rmtree(eval_dir, ignore_errors=True)
    Path(eval_dir).mkdir(parents=True, exist_ok=True)

    # Get random seeds
    eval_eps = 10
    seeds = [random.randint(0, 10000) for _ in range(eval_eps)]

    with open(eval_dir + "/seeds.txt", "w") as f:
        f.write(str(seeds))

    model_path = f"{save_dir}/final_model"
    env = get_env(config["env_name"], config["mu_type"], config["algorithm"])
    env = PPOEnvWrapper(env)
    agent = get_alg(env, config, device)
    agent.load(model_path)
    agent.set_random_seed(0)
    env.set_agent(agent)

    evaluate(
        env=env,
        agent=agent,
        seeds=seeds,
        eval_dir=eval_dir,
        device=device,
    )

    end = time.time()
    with open("log.txt", "a") as f:
        f.write("\n-------------------------------------------------------------\n")
        f.write(f"End time: {time.strftime('%H:%M:%S', time.localtime(end))}\n")
        f.write(f"Took: {(end - start) / 60} minutes\n")
        f.write(f"Config: {config}\n\n")
        f.write("\n-------------------------------------------------------------\n")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--env_name", type=str, default="fico")
    args.add_argument("--algorithm", type=str, default="ppo")
    args.add_argument("--mu_type", type=str, default="accuracy")
    args.add_argument("--train_timesteps", type=int, default=500_000)
    args.add_argument("--n_jobs", type=int, default=1)
    args = args.parse_args()

    # load config
    base_config = OmegaConf.load("configs/base.yaml")
    algo_config = OmegaConf.load(f"configs/{args.algorithm}.yaml")
    config = OmegaConf.merge(base_config, algo_config)

    # create config list for multiprocessing
    config_list = []
    for i, params in enumerate(config.algorithm_param_list):
        params_info = " ".join([f"{k}:{v}" for k, v in params.items()])
        params = OmegaConf.merge(params, config.algorithm_param)
        exp_name = args.algorithm + f"({params_info})"
        config_i = {
            "exp_name": exp_name,
            "env_name": args.env_name,
            "algorithm": args.algorithm,
            "mu_type": args.mu_type,
            "train_timesteps": args.train_timesteps,
            "omega": 0.05,
            "algorithm_params": params,
        }
        config_list.append(config_i)

    mp.set_start_method("spawn")
    processes = []
    for config in config_list:
        p = mp.Process(target=main, args=(config,))
        p.start()
        processes.append(p)
        if len(processes) >= args.n_jobs:
            for p in processes:
                p.join()
            processes = []

    for p in processes:
        p.join()

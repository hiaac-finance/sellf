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

from fairrl.agents.ppo_wrapper_env import PPOEnvWrapper
from fairrl.agents.sellf import SELLF

from fairrl.environments.resampling import (
    ResamplingEnv,
    LendingEnv,
    EnemEnv,
)
import argparse
from omegaconf import OmegaConf


def get_env(env_name: str, utility_method: str, seed: int) -> ResamplingEnv:
    delta_method = "imputation"
    if "fico" in env_name or "setting" in env_name:
        env = LendingEnv(
            utility_method=utility_method,
            delta_method=delta_method,
            distributions=env_name,
            n_applicants=4_000,
            seed=seed,
        )
    elif env_name == "enem":
        env = EnemEnv(
            utility_method=utility_method, delta_method=delta_method, seed=seed
        )
    return env


def get_alg(env, config, device):
    model = SELLF(
        env=env,
        policy_kwargs={
            "use_predictor": config["use_predictor"],
        },
        omega=config["omega"],
        device=device,
        **config["algorithm_params"],
    )
    return model


def train(train_timesteps, env, save_dir, config, device, seed):

    env = PPOEnvWrapper(env=env)
    env = Monitor(env)

    model = get_alg(env, config, device)
    model.set_random_seed(seed)
    env.set_agent(model)

    shutil.rmtree(save_dir, ignore_errors=True)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    model.set_logger(configure(folder=save_dir, format_strings=["csv"]))
    model.learn(total_timesteps=train_timesteps)
    model.save(save_dir + "/final_model")


def plot_learning(exp_dir):
    df_log = pd.read_csv(f"{exp_dir}/progress.csv")

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
    plt.savefig(f"{exp_dir}/learning.png")


def evaluate(env, agent, seeds, eval_dir, device):
    eval_data = []
    num_eps = len(seeds)
    for ep in range(num_eps):
        random.seed(seeds[ep])
        np.random.seed(seeds[ep])
        torch.manual_seed(seeds[ep])
        env.seed(seeds[ep])

        obs = env.reset()
        done = False
        t = 0
        while not done:
            obs = np.array(obs).reshape(1, -1)
            obs = torch.tensor(obs, dtype=torch.float32).to(device)
            action = agent.get_action(obs)
            action = action.item()
            pred = agent.get_label(obs).item()

            obs, _, done, infos = env.step(action)
            resource = env.resource
            eval_data.append(
                {
                    "ep": ep,
                    "t": t,
                    "group_id": infos["group"],
                    "action": action,
                    "label": infos["label"],
                    "pred": pred,
                    "correct": int(infos["label"] == pred),
                    "resource": resource,
                    "delta": infos["delta"],
                    "delta_obs": infos["delta_obs"],
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
    print("-------------------------------------------------------------")
    print(f"Start time: {time.strftime('%H:%M:%S', time.localtime(start))}")
    print(f"Config: {config}")
    print("-------------------------------------------------------------")

    exp_dir = f"./experiments/ablation_2/{config['seed']}/{config['env_name']}/{config['mu_type']}/{config['exp_name']}"
    Path(exp_dir).mkdir(parents=True, exist_ok=True)
    save_dir = f"{exp_dir}/models"
    eval_dir = f"{exp_dir}/eval"
    config["use_predictor"] = config["algorithm"].find("sellf") != -1
    env = get_env(config["env_name"], config["mu_type"], config["seed"])

    # set seeds
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    train(
        train_timesteps=config["train_timesteps"],
        env=env,
        save_dir=save_dir,
        config=config,
        device=device,
        seed=config["seed"],
    )

    plot_learning(save_dir)

    # Initialize eval directory to store eval information
    shutil.rmtree(eval_dir, ignore_errors=True)
    Path(eval_dir).mkdir(parents=True, exist_ok=True)

    # Get random seeds
    seeds = [random.randint(0, 10000) for _ in range(10)]

    with open(eval_dir + "/seeds.txt", "w") as f:
        f.write(str(seeds))

    model_path = f"{save_dir}/final_model"
    env = get_env(config["env_name"], config["mu_type"], config["seed"])
    env = PPOEnvWrapper(env)
    agent = get_alg(env, config, device)
    agent.load(model_path)
    agent.set_random_seed(config["seed"])
    env.set_agent(agent)

    evaluate(
        env=env,
        agent=agent,
        seeds=seeds,
        eval_dir=eval_dir,
        device=device,
    )

    end = time.time()
    print("-------------------------------------------------------------")
    print(f"End time: {time.strftime('%H:%M:%S', time.localtime(end))}")
    print(f"Took: {(end - start) / 60} minutes")
    print(f"Config: {config}")
    print("-------------------------------------------------------------")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--env_name", type=str, default="fico")
    args.add_argument("--algorithm", type=str, default="ppo")
    args.add_argument("--mu_type", type=str, default="tpr")
    args.add_argument("--train_timesteps", type=int, default=500_000)
    args.add_argument("--seed_id", type=int, default=0)
    args = args.parse_args()

    seed_list = [
        0
    ]
    if args.seed_id < len(seed_list):

        # load config
        base_config = OmegaConf.load("configs/base.yaml")
        algo_config = OmegaConf.load(f"configs/sellf_ablation_2.yaml")
        config = OmegaConf.merge(base_config, algo_config)

        for i, params in enumerate(config.algorithm_param_list):
            params_info = " ".join([f"{k}:{v}" for k, v in params.items()])
            params = OmegaConf.merge(config.algorithm_param, params)
            exp_name = args.algorithm + f"({params_info})"
            config_i = {
                "exp_name": exp_name,
                "env_name": args.env_name,
                "algorithm": args.algorithm,
                "mu_type": args.mu_type,
                "train_timesteps": args.train_timesteps,
                "omega": 0.05,
                "algorithm_params": params,
                "seed": seed_list[args.seed_id],
            }
            main(config_i)

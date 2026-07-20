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

from main import get_env, get_alg, evaluate, plot_learning


def train(train_timesteps, env, save_dir, config, device, seed):

    env = PPOEnvWrapper(env=env)
    env = Monitor(env)

    model = get_alg(env, config, device)
    model.calculate_error = True
    model.set_random_seed(seed)
    env.set_agent(model)

    shutil.rmtree(save_dir, ignore_errors=True)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    model.set_logger(configure(folder=save_dir, format_strings=["csv"]))
    model.learn(total_timesteps=train_timesteps)
    model.save(save_dir + "/final_model")


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

    exp_dir = f"./experiments/bound/{config['seed']}/{config['env_name']}/{config['mu_type']}/{config['exp_name']}"
    Path(exp_dir).mkdir(parents=True, exist_ok=True)
    save_dir = f"{exp_dir}/models"
    eval_dir = f"{exp_dir}/eval"
    config["use_predictor"] = config["algorithm"].find("sellf") != -1
    env = get_env(config["env_name"], config["mu_type"], "sellf", config["seed"])

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

    plot_learning(config["env_name"], config["mu_type"], config["exp_name"])

    # Initialize eval directory to store eval information
    shutil.rmtree(eval_dir, ignore_errors=True)
    Path(eval_dir).mkdir(parents=True, exist_ok=True)

    # Get random seeds
    seeds = [config["seed"]]

    with open(eval_dir + "/seeds.txt", "w") as f:
        f.write(str(seeds))

    model_path = f"{save_dir}/final_model"
    env = get_env(config["env_name"], config["mu_type"], config["seed"])
    env = PPOEnvWrapper(env)
    agent = get_alg(env, config, device)
    agent.calculate_error = True
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
    args.add_argument("--mu_type", type=str, default="accuracy")
    args.add_argument("--train_timesteps", type=int, default=500_000)
    args.add_argument("--seed_id", type=int, default=0)
    args = args.parse_args()

    seed_list = [
        8525,
        2892,
        5666,
        3309,
        6898,
        2052,
        8530,
        8624,
        7070,
        8094,
        1961,
        2866,
        9674,
        9684,
        9928,
        9664,
        8616,
        2029,
        9816,
        8070,
        3413,
        2989,
        2807,
        4163,
        7576,
    ]

    # load config
    base_config = OmegaConf.load("configs/base.yaml")
    algo_config = OmegaConf.load(f"configs/sellf_bound.yaml")
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

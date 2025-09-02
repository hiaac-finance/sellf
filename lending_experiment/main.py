from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import shutil
from pathlib import Path
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import sys

sys.path.append("..")

from lending_experiment.agents.ppo_wrapper_env import PPOEnvWrapper
from lending_experiment.agents.ppo import PPO
from lending_experiment.agents.pocar import POCAR
from lending_experiment.agents.rrm import RRM
from lending_experiment.agents.sellf import SELLF

from lending_experiment.environments.resampling import (
    ResamplingEnv,
    LendingEnv,
    EnemEnv,
)
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("Using device: ", device)
torch.cuda.empty_cache()

EXP_DIR = "./experiments"

ALG_PARAMS = {}
ALG_PARAMS["ppo"] = {"learning_rate": 1e-5}
ALG_PARAMS["sellf"] = {
    "learning_rate": 1e-5,
    "beta_0": 1,
    "beta_1": 1.,
    "beta_2": 1.,
    "beta_3": 0.5,
}
ALG_PARAMS["pocar_full"] = {
    "learning_rate": 1e-5,
    "beta_0": 1,
    "beta_1": 0.5,
    "beta_2": 0.5,
}
ALG_PARAMS["pocar"] = {
    "learning_rate": 1e-5,
    "beta_0": 1,
    "beta_1": 0.5,
    "beta_1": 0.5,
}
ALG_PARAMS["rrm"] = {
    "learning_rate": 1e-5,
    "beta_0": 0.5,
}


def get_env(env_name: str, utility_method: str, algorithm: str) -> ResamplingEnv:
    if algorithm == "pocar_full":
        delta_method = "full"
    elif algorithm.find("sellf") != -1:
        delta_method = "imputation"
    else:
        delta_method = "accepted"
    if env_name == "fico":
        env = LendingEnv(utility_method=utility_method, delta_method=delta_method)
    elif env_name == "fico_equal":
        env = LendingEnv(
            utility_method=utility_method,
            delta_method=delta_method,
            group_ratios="equal",
        )
    elif env_name == "enem":
        env = EnemEnv(
            n_features=130, utility_method=utility_method, delta_method=delta_method
        )
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

    return model


def train(train_timesteps, env, save_dir, config):

    env = PPOEnvWrapper(env=env)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    model = get_alg(env, config, device)
    env.env_method("set_agent", model)

    shutil.rmtree(save_dir, ignore_errors=True)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    model.set_logger(configure(folder=save_dir))
    model.learn(total_timesteps=train_timesteps)
    model.save(save_dir + "/final_model")


def evaluate(env, agent, seeds, eval_dir):
    eval_data = []
    num_eps = len(seeds)
    for ep in range(num_eps):
        random.seed(seeds[ep])
        np.random.seed(seeds[ep])
        torch.manual_seed(seeds[ep])

        env.update_models()

        obs = env.reset()
        done = False
        t = 0
        while not done:
            obs = np.array(obs).reshape(1, -1)
            obs = torch.tensor(obs, dtype=torch.float32).to(device)
            action = agent.policy.get_action(obs)
            action = action.item()
            pred = agent.policy.get_label(obs).item()

            applicant = env.pool[env.idx]
            # Logging
            group_id = np.argmax(applicant["group"])
            # Add to loans if the agent wants to loan
            label = applicant["label"]
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
                    "delta_real": env.delta_real,
                }
            )
            t += 1
            if done:
                break

    eval_data = pd.DataFrame(eval_data)
    eval_data.to_csv(f"{eval_dir}/eval_data.csv", index=False)

    return eval_data


def main(config):
    start = time.time()
    with open("log.txt", "a") as f:
        f.write("\n-------------------------------------------------------------\n")
        f.write(f"Start time: {time.strftime('%H:%M:%S', time.localtime(start))}\n")
        f.write(f"Environment: {config['env_name']}\n")
        f.write(f"Algorithm: {config['algorithm']}\n")
        f.write(f"Train Timesteps: {config['train_timesteps']}\n")

    exp_dir = (
        f"./experiments/{config['env_name']}/{config['mu_type']}/{config['exp_name']}"
    )
    save_dir = f"{exp_dir}/models"
    eval_dir = f"{exp_dir}/eval"
    config["use_predictor"] = config["algorithm"].find("sellf") != -1
    env = get_env(config["env_name"], config["mu_type"], config["algorithm"])
    train(
        train_timesteps=config["train_timesteps"],
        env=env,
        save_dir=save_dir,
        config=config,
    )

    # Initialize eval directory to store eval information
    shutil.rmtree(eval_dir, ignore_errors=True)
    Path(eval_dir).mkdir(parents=True, exist_ok=True)

    # Get random seeds
    eval_eps = 5
    seeds = [random.randint(0, 10000) for _ in range(eval_eps)]

    with open(eval_dir + "/seeds.txt", "w") as f:
        f.write(str(seeds))

    model_path = f"{save_dir}/final_model"
    env = get_env(config["env_name"], config["mu_type"], config["algorithm"])
    env = PPOEnvWrapper(env)
    agent = get_alg(env, config, device)
    agent.load(model_path)
    env.set_agent(agent)

    evaluate(
        env=env,
        agent=agent,
        seeds=seeds,
        eval_dir=eval_dir,
    )

    end = time.time()
    with open("log.txt", "a") as f:
        f.write(f"End time: {time.strftime('%H:%M:%S', time.localtime(end))}\n")
        f.write(f"Took: {(end - start) / 60} minutes\n")
        f.write("\n")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    # params are the env_name, the algorithm, and the mu_type
    args.add_argument("--env_name", type=str, default="fico")
    args.add_argument("--algorithm", type=str, default="ppo")
    args.add_argument("--mu_type", type=str, default="accuracy")
    args = args.parse_args()

    train_timesteps = 500_000
    config = {
        "exp_name": args.algorithm,
        "env_name": args.env_name,
        "algorithm": args.algorithm,
        "train_timesteps": train_timesteps,
        "mu_type": args.mu_type,
        "omega": 0.05,
        "algorithm_params": ALG_PARAMS[args.algorithm],
    }
    main(config)

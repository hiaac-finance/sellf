from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import shutil
from pathlib import Path
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

# from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import sys

sys.path.append("..")

from lending_experiment.agents.ppo.ppo_wrapper_env import PPOEnvWrapper, Monitor
from lending_experiment.agents.ppo.sb3.ppo import PPO

from lending_experiment.environments.resampling import (
    ResamplingEnv,
    LendingEnv,
    EnemEnv,
    Params as ResamplingParams,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)
torch.cuda.empty_cache()

EXP_DIR = "./experiments"


def get_env(env_name: str, utility_method: str, algorithm: str) -> ResamplingEnv:
    if algorithm == "pocar_full":
        delta_method = "full"
    elif algorithm == "sellf":
        delta_method = "imputation"
    else:
        delta_method = "accepted"
    if env_name == "fico":
        params = ResamplingParams()
        env = LendingEnv(
            params, utility_method=utility_method, delta_method=delta_method
        )
    elif env_name == "enem":
        params = ResamplingParams()
        params.num_features = 130
        env = EnemEnv()
    return env


def train(train_timesteps, env, save_dir, config):

    # print("env_params: ", env.state.params)

    env = PPOEnvWrapper(env=env)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    use_predictor = config["algorithm"] == "sellf"

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs={
            "use_predictor": use_predictor,
            "activation_fn": torch.nn.ReLU,
            "net_arch": [256, 256, dict(vf=[256, 128], pi=[256, 128])],
        },
        verbose=0,
        omega=config["omega"],
        device=device,
        **config["algorithm_params"],
    )
    env.env_method("set_agent", model)

    shutil.rmtree(save_dir, ignore_errors=True)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000, save_path=save_dir, name_prefix="rl_model"
    )

    model.set_logger(configure(folder=save_dir))
    model.learn(total_timesteps=train_timesteps, callback=checkpoint_callback)
    model.save(save_dir + "/final_model")

    # Once we finish learning, plot the returns over time and save into the experiments directory
    # plot_rets(EXP_DIR)


def evaluate(env, agent, seeds, eval_dir):
    eval_data = []
    num_eps = len(seeds)
    for ep in range(num_eps):
        random.seed(seeds[ep])
        np.random.seed(seeds[ep])
        torch.manual_seed(seeds[ep])

        # Make predictions for everyone
        action_list = []
        pred_list = []
        acc = []
        for idx in range(env.num_applicants):
            obs = env.get_applicant_obs(idx)
            obs = np.array(obs).reshape(1, -1)
            action = agent.predict(obs)[0]
            obs = torch.tensor(obs, dtype=torch.float32).to(device)
            pred = agent.policy.predict_label(obs).item()
            action_list.append(action)
            pred_list.append(pred)
            label = env.pool[idx]["label"]
            acc.append(int(label == pred))
        print(f"Mean accuracy: {np.mean(acc)}")

        env.set_action_pred(action_list, pred_list)

        obs = env.reset()
        done = False
        t = 0
        while not done:
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
            label = env.state.label
            obs, _, done, _ = env.step(action)
            resource = env.state.resource
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
                    "delta": env.state.delta,
                    "delta_real": env.state.delta_real,
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
        f"./experiments/{config['env_name']}/{config['mu_type']}/{config['algorithm']}"
    )
    save_dir = f"{exp_dir}/models"
    eval_dir = f"{exp_dir}/eval"
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

    model_path = f"{save_dir}/final_model.zip"
    agent = PPO.load(model_path, verbose=1)
    env = get_env(config["env_name"], config["mu_type"], config["algorithm"])
    env = PPOEnvWrapper(env)
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
    env_name = "fico"
    train_timesteps = 250_000
    algorithms_params = {
        "sellf" :{
            "ad_reg": "sellf",
            "learning_rate": 1e-5,
            "beta_0": 1,
            "beta_1": 5,
            "beta_3": 3,
            "bound_type": "var_up",
        },
        "sellf1" :{
            "ad_reg": "sellf",
            "learning_rate": 1e-5,
            "beta_0": 1,
            "beta_1": 5,
            "beta_3": 3,
            "bound_type": "diff",
        },
        "sellf2" :{
            "ad_reg": "sellf",
            "learning_rate": 1e-5,
            "beta_0": 1,
            "beta_1": 5,
            "beta_3": 3,
            "bound_type": "var",
        },
    }

    for algo, algo_params in algorithms_params.items():
        config = {
            "env_name": "fico",
            "algorithm": algo,
            "train_timesteps": train_timesteps,
            "mu_type": "accuracy",
            "omega": 0.05,
            "algorithm_params": algo_params,
        }

        main(config)

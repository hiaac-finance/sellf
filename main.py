import argparse
import os
import zipfile
from datetime import datetime
import random
import shutil
from pathlib import Path
from copy import deepcopy
from collections import deque

import numpy as np
import pickle
import torch
import tqdm
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

from agents.ppo.q_custom_utils import QMonitor as Monitor
from agents.ppo.q_custom_utils import QDummyVecEnv as DummyVecEnv

from config import Config
from environments.lending import DelayedImpactEnv
from environments.lending_params import DelayedImpactParams, two_group_credit_clusters
from environments.rewards import LendingReward
from agents.ppo.ppo_wrapper_env import PPOEnvWrapper
from agents.ppo.sb3.ppo import PPO
from agents.ppo.sb3.new_rollout_buff import DummyEvalBuffer, RolloutBuffer
from graphing.plot_single import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device: ', device)
torch.cuda.empty_cache()

config = Config()

random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
# torch.cuda.manual_seed(SEED)

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


def train_multi(train_timesteps, env_params, config_params, device, should_load=False):
    EXP_DIR = config_params.EXP_DIR
    SAVE_DIR = config_params.SAVE_DIR
    SAVE_FREQ = config_params.SAVE_FREQ

    print('env_params: ', env_params)

    model = None
    
    shutil.rmtree(EXP_DIR, ignore_errors=True)
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

    save_code(config_params.EXP_DIR)

    with open(os.path.join(SAVE_DIR, 'config.txt'), 'w') as f:
        f.write('----------------------------------------------------\n')
        f.write('----------------------------------------------------\n')
        date_time_str = datetime.now().strftime("%b %d, %Y at %H:%M:%S")
        f.write(date_time_str + '\n\n')
        keys = [key for key in dir(config_params) if not key.startswith('__')]
        for key in keys:
            f.write(f'{key}: {getattr(config_params,key)}\n')

        f.write(f'\n')

    seeds = [random.randint(0, 10000) for i in range(config_params.NUM_T_SEEDS)]

    for i in range(config_params.NUM_T_SEEDS):
        env = DelayedImpactEnv(env_params)
        seed = seeds[i]
        env.seed(seed)
        
        env = PPOEnvWrapper(env=env, reward_fn=LendingReward, config_params=config_params)
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        
        with open(os.path.join(SAVE_DIR, 'config.txt'), 'a') as f:
            f.write(f'SEED{i}: {seed}\n')
        
        save_dir = os.path.join(SAVE_DIR, f'seed_{seed}')

        model = PPO("MlpPolicy", env, device=device, config_params=config_params)

        checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ, save_path=save_dir,
                                                name_prefix='rl_model')

        model.set_logger(configure(folder=save_dir, format_strings=['log', 'csv']))
        
        model.learn(total_timesteps=train_timesteps, callback=checkpoint_callback)
        model.save(os.path.join(save_dir, 'final_model'))

        # Once we finish learning, plot the returns over time and save into the experiments directory
        try:
            plot_rets(save_dir)
        except:
            plt.close()
            print(f'Could not plot returns for {save_dir}')

        column_list = [ 'train/cumulative_gx_g0', 'train/cumulative_gx_g1',
                    'rollout/ep_rew_mean', 'train/policy_constraint_loss', 'train/lambda_loss',
            'train/policy_gradient_loss', 'train/value_loss',
            'train/static_kl', 'train/cumulative_reward', 'train/soft_de', 'train/soft_ie', 'train/soft_se', 'train/c_pi_theta',
            'train/rolling_soft_de', 'train/rolling_soft_ie', 'train/rolling_soft_se', 'train/rolling_c_pi_theta']
        try:
            plot_progress_data(save_dir, column_list)
        except:
            plt.close()
            print(f'Could not plot progress data for {save_dir}')

    try:
        plot_multi_seed_progress_data(SAVE_DIR, seeds, column_list)
    except:
        plt.close()
        print(f'Could not plot multi_seed_progress_data for {SAVE_DIR}')
    return seeds
    


# def evaluate(env, agent, num_eps, num_timesteps, name, seeds, eval_path, config_params, algorithm=None):
def evaluate(env, agent, num_eps, name, seeds, eval_path, config_params, algorithm=None):
    EVAL_ZETA_0 = config_params.EVAL_ZETA_0
    EVAL_ZETA_1 = config_params.EVAL_ZETA_1
    NUM_GROUPS = config_params.NUM_GROUPS
    WINDOW = config_params.WINDOW

    num_timesteps = config_params.EVAL_EP_TIMESTEPS
    num_cscores = len(env.state.params.applicant_distribution.components[0].weights)
    max_cscore = num_cscores - 1

    print(f"Evaluating {name}")
    eval_data = {
        'tot_bank_cash_over_time': np.zeros((num_eps, num_timesteps)),  # The amount of bank cash per timestep per episode
        'tot_g_sum': np.zeros((num_eps, num_timesteps, NUM_GROUPS)), # total qualification gain by group per timestep per episode
        'tot_loans_over_time_by_cscore': np.zeros((num_eps, num_timesteps,  NUM_GROUPS, num_cscores)), 
        'tot_cscore_seen_over_time': np.zeros((num_eps, num_timesteps,  NUM_GROUPS, num_cscores)),
        'tot_tpr_over_time' : np.zeros((num_eps, num_timesteps, NUM_GROUPS)),
        'tot_tpr_obs_over_time' : np.zeros((num_eps, num_timesteps, NUM_GROUPS)),
        "tot_acc_over_time" : np.zeros((num_eps, num_timesteps, NUM_GROUPS)),
        "tot_acc_obs_over_time" : np.zeros((num_eps, num_timesteps, NUM_GROUPS)),        
        # -------------------------------------------------------------------
    }

    reward_fn = LendingReward(config_params.OMEGA)

    cscore_tensor = torch.zeros((num_cscores, num_cscores)).to(agent.device)
    for i in range(num_cscores):
        cscore_tensor[i, i] = 1

    for ep in range(num_eps):
        random.seed(seeds[ep])
        np.random.seed(seeds[ep])
        torch.manual_seed(seeds[ep])

        env.seed(seeds[ep])

        obs = env.reset()
        
        done = False
        g_sum = {0: 0, 1: 0}
        print(f'Episode {ep}:')
        loans_ot_by_cscore = np.zeros((NUM_GROUPS, num_cscores))
        cscore_seen_over_time = np.zeros((NUM_GROUPS, num_cscores))
        dummy_buff = DummyEvalBuffer(num_timesteps, obs.shape, 2, 1, 100, config_params.DELAYED_IMPACT_CLUSTER_PROBS, config_params.QUAL_CHANGE)
        for t in tqdm.trange(num_timesteps):

            action = None
            if algorithm == 'cpo':
                action = int(agent(torch.FloatTensor(obs).squeeze()).sample().item())
            else:
                if isinstance(agent, PPO):
                    action = agent.predict(obs)[0]
                else:
                    action = agent.act(obs, done)

            # Logging
            group_id = np.argmax(env.state.group)

            c_score_idx = np.min(obs.nonzero())
            curr_x = c_score_idx
            cscore_seen_over_time[group_id][c_score_idx] += 1

            eval_data['tot_cscore_seen_over_time'][ep][t] = cscore_seen_over_time
            # Add to loans if the agent wants to loan
            if action == 1:
                loans_ot_by_cscore[group_id][c_score_idx] += 1

            next_x = env.next_x_given_action(action)

            old_bank_cash = env.state.bank_cash

            # --- NEW ---
            eval_data['tot_tpr_over_time'][ep][t] = env.tpr
            eval_data['tot_tpr_obs_over_time'][ep][t] = env.tpr_obs
            eval_data['tot_acc_over_time'][ep][t] = env.acc
            eval_data['tot_acc_obs_over_time'][ep][t] = env.acc_obs

            benefit_delta = agent.benefit_deltas_dict[group_id][curr_x]

            next_obs, rew, done, info, gx, gx_pi0, action_pi0, next_x_pi0 = env.step(action)

            dummy_buff.add(obs, action, gx, curr_x, next_x, gx_pi0, action_pi0, next_x_pi0, benefit_delta)

            g_sum[group_id] += gx
            eval_data['tot_g_sum'][ep][t][0], eval_data['tot_g_sum'][ep][t][1] = g_sum[0], g_sum[1]

            eval_data['tot_loans_over_time_by_cscore'][ep][t] = loans_ot_by_cscore
            bank_cash = env.state.bank_cash

            eval_data['tot_bank_cash_over_time'][ep][t] = bank_cash
            obs = next_obs
            if done:
                break

    # ------------------- new part -------------------------
    if not os.path.isdir(eval_path):
        os.makedirs(eval_path, exist_ok=True)

    with open(os.path.join(eval_path, 'tot_eval_data.pkl'), 'wb') as f:
        pickle.dump(eval_data, f)
    # ------------------------------------------------------

    return eval_data

def plot_figures(eval_path):
    with open(os.path.join(eval_path, 'tot_eval_data.pkl'), 'rb') as f:
        eval_data = pickle.load(f)

    plot_bank_cash_over_time(eval_data, eval_path)
    plot_g_by_group_over_time(eval_data, eval_path)
    plot_g_by_group_over_time(eval_data, eval_path, "tpr_over_time")
    plot_g_by_group_over_time(eval_data, eval_path, "tpr_obs_over_time")
    plot_g_by_group_over_time(eval_data, eval_path, "acc_over_time")
    plot_g_by_group_over_time(eval_data, eval_path, "acc_obs_over_time")
    plot_g_disparity_over_time(eval_data, eval_path, "g_sum")


def main(conf=config, arg_train=False, arg_eval=False, is_outside_func_call=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['ppo'])
    parser.add_argument('--eval_path', dest='eval_path', type=str, default=conf.EVAL_DIR)
    args = parser.parse_args()


    if is_outside_func_call:
        print('Is outside function call...')
        random.seed(conf.SEED)
        np.random.seed(conf.SEED)
        torch.manual_seed(conf.SEED)


    if conf.BETA_C_PI != 0.0:
        assert(conf.REGULARIZE_ADVANTAGE == False)
    if conf.REGULARIZE_ADVANTAGE == True:
        assert(conf.BETA_C_PI == 0.0)

    env_params = DelayedImpactParams(
        applicant_distribution=two_group_credit_clusters(
            cluster_probabilities=conf.CLUSTER_PROBABILITIES,
            group_likelihoods=[conf.GROUP_0_PROB, 1 - conf.GROUP_0_PROB],
            success_probabilities=conf.DELAYED_IMPACT_SUCCESS_PROBS,
            credit_drift_probs=conf.DRIFT_PROBS,),
        bank_starting_cash=conf.BANK_STARTING_CASH,
        interest_rate=conf.INTEREST_RATE,
        cluster_shift_increment=conf.CLUSTER_SHIFT_INCREMENT,
    )
    env = DelayedImpactEnv(env_params)
    env.seed(conf.SEED)
        
    # make sure nothing gets overwritten for training
    should_load = False
    if args.train or arg_train:
        exp_exists = False
        if os.path.isdir(conf.SAVE_DIR):
            exp_exists = True
            if input(f'{conf.SAVE_DIR} already exists; do you want to retrain / continue training? (y/n): ') != 'y':
                exit()

            print('Training from start...')
        if exp_exists:
            resp = input(f'\nWould you like to load the previous model to continue training? If you do not select yes, you will start a new training. (y/n): ')
            if resp != 'y' and resp != 'n':
                exit('Invalid response for resp: ' + resp)
            should_load = resp == 'y'
            
    if args.eval or arg_eval:
        if os.path.isdir(conf.EVAL_DIR):
            if input(f'{conf.EVAL_DIR} already exists; do you want to overwrite? (y/n): ') != 'y':
                print('Please update the EVAL_DIR in config.py to a new directory to store the evaluation results.')
                print('Exiting...')
                exit()

    t_seeds = None
    if args.train or arg_train:
        # if conf.NUM_T_SEEDS != 1:
        t_seeds = train_multi(train_timesteps=conf.TRAIN_TIMESTEPS, env_params=env_params, config_params=conf, device=device, should_load=should_load)

    if args.eval or arg_eval: 
        # Initialize eval directory to store eval information
        shutil.rmtree(conf.EVAL_DIR, ignore_errors=True)
        Path(conf.EVAL_DIR).mkdir(parents=True, exist_ok=True)

        # Get random seeds
        eval_eps = 5 # 10
        eval_timesteps = conf.EVAL_EP_TIMESTEPS
        seeds = [random.randint(0, 10000) for _ in range(eval_eps)]

        with open(os.path.join(conf.EVAL_DIR, 'seeds.txt'), 'w') as f:
            f.write(str(seeds)+"\n")
            f.write(str(conf))

        eval_paths = []

        for name, model_path in conf.EVAL_MODEL_PATHS.items():
            if conf.NUM_T_SEEDS != 1:
                weights_step = model_path.split('/')[-1]
                base_path = conf.SAVE_DIR
                seed_dirs = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

                for seed_dir in seed_dirs:
                    model_path = os.path.join(base_path, seed_dir, weights_step)
                    env = DelayedImpactEnv(env_params)
                    agent = PPO.load(model_path, device=device, verbose=1)

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
                env = DelayedImpactEnv(env_params)
                agent = PPO.load(model_path, verbose=1)
                
                evaluate(env=PPOEnvWrapper(env=env, reward_fn=LendingReward, config_params=conf, is_eval=True),
                        agent=agent,
                        num_eps=eval_eps,
                        # num_timesteps=eval_timesteps,
                        name=name,
                        seeds=seeds,
                        eval_path=os.path.join(args.eval_path, name),
                            config_params=conf,
                    )
                eval_paths.append(os.path.join(args.eval_path, name))

        for path in eval_paths:
            plot_figures(path)

    
    if is_outside_func_call:
        return f'Finished running experiment for {conf.MODEL}'



if __name__ == '__main__':
    main()
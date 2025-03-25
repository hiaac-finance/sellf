import os

import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from geomloss import SamplesLoss
wloss = SamplesLoss("sinkhorn", p=1, blur=0.01)

"""
For plotting a single model's results.
"""

def plot_bank_cash_over_time(tot_eval_data, path):
    aggregated_tot_ep_bank_cash = tot_eval_data['tot_bank_cash_over_time']

    timesteps = list(range(aggregated_tot_ep_bank_cash.shape[1]))
    means = np.mean(aggregated_tot_ep_bank_cash, axis=0) # Shape: (num human_designed_policies, num timesteps)

    plt.plot(timesteps, means, alpha=0.8)
    plt.title(f'Average Bank Cash Over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Bank Cash')
    plt.savefig(os.path.join(path, 'bank_cash_over_time.png'))
    plt.close()

def plot_rets(exp_path, save_png=True):
    if not os.path.isdir(exp_path):
        exit(f"{exp_path} not found!!!")

    df = pd.read_csv(os.path.join(exp_path,'progress.csv'))
    xs = df['time/total_timesteps']
    ys = df['rollout/ep_rew_mean']

    plt.plot(xs.values, ys.values)
    plt.title('PPO Training: Average Episodic Return Over Time')
    plt.xlabel('Total Timesteps Trained So Far')
    plt.ylabel('Average Episodic Return')
    if save_png:
        plt.savefig(os.path.join(exp_path,'train_ret_over_time'))
    else:
        plt.show()

    plt.close()

def plot_g_over_time(tot_eval_data, path, g_type='g_sum'):
    d_name = 'tot_' + g_type
    tot_g_over_time = np.mean(tot_eval_data[d_name], axis=0)
    g_std = np.std(tot_eval_data[d_name], axis=0)
    timesteps = np.arange(tot_g_over_time.shape[0])
    plt.plot(timesteps, np.sum(tot_g_over_time, axis=1))
    plt.fill_between(timesteps, np.sum(tot_g_over_time, axis=1) - np.sum(g_std, axis=1), np.sum(tot_g_over_time, axis=1) + np.sum(g_std, axis=1), alpha=0.4)

    plt.xlabel('Timestep (t)', fontsize=15)
    if g_type == 'g_pi0_sum':
        g = "g'"
        policy = 'pi_0'
    else:
        g = "g"
        policy = 'pi'
    plt.ylabel(f"Qual change ({g})", fontsize=15)
    plt.title(f"Total Qualification Gain ({g}, {policy}) over time", fontsize=15)
    # plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(path, f'tot_{g_type}over_time.png'))
    plt.close()

def plot_g_by_group_over_time(tot_eval_data, path, g_type='g_sum'):
    d_name = 'tot_' + g_type
    tot_g_over_time = np.mean(tot_eval_data[d_name], axis=0)
    tot_g_std = np.std(tot_eval_data[d_name], axis=0)
    timesteps = np.arange(tot_g_over_time.shape[0])

    plt.plot(timesteps, tot_g_over_time[:,0], label=f'G1')
    plt.fill_between(timesteps, tot_g_over_time[:,0] - tot_g_std[:,0], tot_g_over_time[:,0] + tot_g_std[:,0], alpha=0.4)
    plt.plot(timesteps, tot_g_over_time[:,1], label=f'G2', linestyle='--', alpha=0.4)
    plt.fill_between(timesteps, tot_g_over_time[:,1] - tot_g_std[:,1], tot_g_over_time[:,1] + tot_g_std[:,1], alpha=0.4)

    plt.xlabel('Timestep (t)', fontsize=15)
    if g_type == 'g_pi0_sum':
        g = "g'"
        policy = 'pi_0'
    else:
        g = "g"
        policy = 'pi'
    #plt.ylabel(f"Qual change ({g})", fontsize=15)
    plt.ylabel(d_name, fontsize=15)
    #plt.title(f"Qualification Gain by Group over time", fontsize=15)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(path, f'tot_{g_type}by_group_over_time.png'))
    plt.close()

def plot_loans_over_time_by_score(tot_eval_data, steps, path):
    width = 0.25
    n_steps = len(steps)
    tot_loans_cscore_dists = np.mean(tot_eval_data['tot_loans_over_time_by_cscore'], axis=0)
    tot_loans_cscore_dists_std = np.std(tot_eval_data['tot_loans_over_time_by_cscore'], axis=0)

    # Create a grid of subplots
    fig, axs = plt.subplots(1, n_steps, figsize=(5 * n_steps, 5))

    # Iterate over each timestep and plot on a separate subplot
    for i, step in enumerate(steps):
        # x = np.array(range(7))
        x = np.array(range(tot_loans_cscore_dists.shape[-1]))
        if n_steps > 1:
            ax = axs[i]
        else:
            ax = axs

        ax.bar(x - width / 2, tot_loans_cscore_dists[step, 0], width, label='Group 1')
        ax.errorbar(x - width / 2, tot_loans_cscore_dists[step, 0], yerr=tot_loans_cscore_dists_std[step, 0], fmt='o', color='black')
        ax.bar(x + width / 2, tot_loans_cscore_dists[step, 1], width, label='Group 2')
        ax.errorbar(x + width / 2, tot_loans_cscore_dists[step, 1], yerr=tot_loans_cscore_dists_std[step, 1], fmt='o', color='black')

        ax.set_title(f'Loans by Cred Score at Step {step}')
        ax.set_xlabel('Credit Score')
        ax.set_ylabel('# Loans')
        ax.legend()

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'cred_score_loans.png'))
    plt.close()

# for each credit score, the fraction of loans given to each group
# ie number of loans given to group 1 with credit score 0 / total number of people with credit score 0
# seen from group 1
# ie loan rate for each group at each credit score level
def plot_frac_loans_over_time_by_score(tot_eval_data, steps, path):
    width = 0.25
    n_steps = len(steps)
    tot_loans_cscore_over_time = np.mean(tot_eval_data['tot_loans_over_time_by_cscore'], axis=0)
    tot_cscore_seen_over_time = np.mean(tot_eval_data['tot_cscore_seen_over_time'], axis=0)

    # Create a grid of subplots
    fig, axs = plt.subplots(1, n_steps, figsize=(5 * n_steps, 5))

    # Iterate over each timestep and plot on a separate subplot
    for i, step in enumerate(steps):
        # x = np.array(range(7))
        x = np.array(range(tot_cscore_seen_over_time.shape[-1]))
        if n_steps > 1:
            ax = axs[i]
        else:
            ax = axs
        g1_frac = tot_loans_cscore_over_time[step, 0] / tot_cscore_seen_over_time[step, 0]
        g2_frac = tot_loans_cscore_over_time[step, 1] / tot_cscore_seen_over_time[step, 1]
        ax.bar(x - width / 2, g1_frac, width, label='Group 1')
        ax.bar(x + width / 2, g2_frac, width, label='Group 2')

        ax.set_title(f'Loan Rate by Cred Score at Step {step}')
        ax.set_xlabel('Credit Score')
        ax.set_ylabel('Proportion of Loans')
        ax.legend()

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'cred_score_loan_rate_by_group.png'))
    plt.close()

def plot_cscore_seen_over_time(tot_eval_data, steps, path):
    width = 0.25
    n_steps = len(steps)
    tot_cscore_seen_over_time = np.mean(tot_eval_data['tot_cscore_seen_over_time'], axis=0)

    # Create a grid of subplots
    fig, axs = plt.subplots(2, n_steps, figsize=(5 * n_steps, 5))

    # Iterate over each timestep and plot on a separate subplot
    for i, step in enumerate(steps):
        # x = np.array(range(7))
        x = np.array(range(tot_cscore_seen_over_time.shape[-1]))
        if n_steps > 1:
            ax = axs[i]
        else:
            ax = axs
        
        ax.bar(x - width / 2, tot_cscore_seen_over_time[step, 0], width, label='Group 1')
        ax.bar(x + width / 2, tot_cscore_seen_over_time[step, 1], width, label='Group 2')

        ax.set_title(f'Sum of Loan Applications by Cred Score at {step}')
        ax.set_xlabel('Credit Score')
        ax.set_ylabel('Number of Applications')
        ax.legend()

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'cred_score_loan_app_sum_by_group.png'))
    plt.close()

def plot_cscore_bar_stats(tot_eval_data, steps, path):
    width = 0.25
    n_steps = len(steps)
    tot_cscore_seen_over_time = np.mean(tot_eval_data['tot_cscore_seen_over_time'], axis=0)
    tot_cscore_seen_over_time_std = np.std(tot_eval_data['tot_cscore_seen_over_time'], axis=0)
    tot_loans_cscore_over_time = np.mean(tot_eval_data['tot_loans_over_time_by_cscore'], axis=0)
    tot_loans_cscore_over_time_std = np.std(tot_eval_data['tot_loans_over_time_by_cscore'], axis=0)
    

    # Create a grid of subplots
    fig, axs = plt.subplots(3, n_steps, figsize=(5 * n_steps, 5*2))
    axs = axs.flatten()
    # Iterate over each timestep and plot on a separate subplot
    for i, step in enumerate(steps):
        # x = np.array(range(7))
        x = np.array(range(tot_cscore_seen_over_time.shape[-1]))
  
        ax = axs[i]
        
        ax.bar(x - width / 2, tot_cscore_seen_over_time[step, 0], width, label='Group 1')
        ax.errorbar(x - width / 2, tot_cscore_seen_over_time[step, 0], yerr=tot_cscore_seen_over_time_std[step, 0], fmt='o', color='black')
        ax.bar(x + width / 2, tot_cscore_seen_over_time[step, 1], width, label='Group 2')
        ax.errorbar(x + width / 2, tot_cscore_seen_over_time[step, 1], yerr=tot_cscore_seen_over_time_std[step, 1], fmt='o', color='black')

        ax.set_title(f'Sum of Loan Applications by Cred Score at {step}')
        ax.set_xlabel('Credit Score')
        ax.set_ylabel('Number of Applications')
        ax.legend()

    for i, step in enumerate(steps):
        # x = np.array(range(7))
        x = np.array(range(tot_cscore_seen_over_time.shape[-1]))
        idx = i + n_steps

        ax = axs[idx]

        ax.bar(x - width / 2, tot_loans_cscore_over_time[step, 0], width, label='Group 1')
        ax.errorbar(x - width / 2, tot_loans_cscore_over_time[step, 0], yerr=tot_loans_cscore_over_time_std[step, 0], fmt='o', color='black')
        ax.bar(x + width / 2, tot_loans_cscore_over_time[step, 1], width, label='Group 2')
        ax.errorbar(x + width / 2, tot_loans_cscore_over_time[step, 1], yerr=tot_loans_cscore_over_time_std[step, 1], fmt='o', color='black')

        ax.set_title(f'Sum of Loans Received by Cred Score at Step {step}')
        ax.set_xlabel('Credit Score')
        ax.set_ylabel('Number Received')
        ax.legend()

    for i, step in enumerate(steps):
        # x = np.array(range(7))
        x = np.array(range(tot_cscore_seen_over_time.shape[-1]))
        idx = i + 2 * n_steps
        ax = axs[idx]
        
        g1_frac = tot_loans_cscore_over_time[step, 0] / tot_cscore_seen_over_time[step, 0]
        g2_frac = tot_loans_cscore_over_time[step, 1] / tot_cscore_seen_over_time[step, 1]
        ax.bar(x - width / 2, g1_frac, width, label='Group 1')
        ax.bar(x + width / 2, g2_frac, width, label='Group 2')

        ax.set_title(f'Loan Rate by Cred Score at Step {step}')
        ax.set_xlabel('Credit Score')
        ax.set_ylabel('Proportion of Loans')
        ax.legend()

    

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'cscore_bar_stats_by_steps.png'))
    plt.close()


def plot_g_disparity_over_time(data, save_dir, g_type='g_sum'):
    d_name = 'tot_' + g_type
    # tot_g_over_time = np.mean(data['tot_g_sum'], axis=0)
    tot_g_over_time = data[d_name]
    g_diff = np.abs(tot_g_over_time[...,0] - tot_g_over_time[...,1])
    g_diff_std = np.std(g_diff, axis=0)
    g_diff = np.mean(g_diff, axis=0)
    # import pdb; pdb.set_trace()
    timesteps = np.arange(g_diff.shape[0])
    # g_diff = np.abs(tot_g_over_time[:,0] - tot_g_over_time[:,1])
    if g_type == 'g_pi0_sum':
        g = "g'"
        policy = 'pi_0'
        # plt.plot(timesteps, g_diff, label=f"(g', pi_0)")
    else:
        g = "g"
        policy = 'pi'
    plt.plot(timesteps, g_diff, label=f"({g}, {policy})")
    plt.fill_between(timesteps, g_diff - g_diff_std, g_diff + g_diff_std, alpha=0.4)
        # plt.plot(timesteps, tot_g_over_time[:,1], c=cmap[name], label=f'G1_{name}', linestyle='--', alpha=0.3)

    # plt.title('Cumulative loans')
    plt.xlabel('Timestep (t)', fontsize=15)
    plt.ylabel(f"Qual. Gain disparity ({g})", fontsize=15)
    plt.title(f"Qualification Gain Disparity ({g}, {policy}) over time", fontsize=15)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'tot_{g_type}_disparity_over_time.png'))
    plt.close()


def plot_progress_data(exp_path, columns):
    # Load each CSV file and store the data for the specified columns

    df = pd.read_csv(os.path.join(exp_path,'progress.csv'))
    missing_columns = set(columns) - set(df.columns) 
    if missing_columns:   
        columns = list(set(columns) - missing_columns)

    num_plots = len(columns)
    n_cols = 2
    n_rows = num_plots // n_cols + 1
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))
    # print(n_rows, n_cols)
    # Plotting the data for each column
    j = 0
    for i, col in enumerate(columns):
        if i % n_cols == 0 and i != 0:
            j += 1
        ax = axs[j, i % n_cols]

        ax.plot(df[col])
        
        ax.set_title(f'Data for: {col}')
        ax.set_xlabel('Index')
        ax.set_ylabel(col)

    plt.tight_layout()
    plt.savefig(os.path.join(exp_path, 'training_plots.png'))
    plt.close()


def plot_multi_seed_progress_data(exp_path, seeds, columns=None):
    dfs = []  # List to store data from each seed
        
    # Loop over each seed to load the respective CSV file
    for seed in seeds:
        # Construct the path to the CSV file for the current seed
        seed_path = os.path.join(exp_path, f'seed_{seed}', 'progress.csv')
        
        # Load the CSV file if it exists
        if os.path.exists(seed_path):
            df = pd.read_csv(seed_path)
            
            # Keep only the specified columns if they exist in this file
            available_columns = set(columns).intersection(set(df.columns))
            if available_columns:
                dfs.append(df[list(available_columns)])
    
    # Check if we successfully loaded any data
    if not dfs:
        print("No data files found or specified columns missing in all files.")
        return
    
    # Concatenate all dataframes along the index
    df_concat = pd.concat(dfs, axis=0)
    
    # Calculate the mean and standard deviation for each column
    df_mean = df_concat.groupby(df_concat.index).mean()
    df_std = df_concat.groupby(df_concat.index).std()

    # Save the averaged and standard deviation data to CSV files
    mean_csv_path = os.path.join(exp_path, 'average_data_wqkl.csv')
    std_csv_path = os.path.join(exp_path, 'std_deviation_data_wqkl.csv')
    
    df_mean.to_csv(mean_csv_path)
    df_std.to_csv(std_csv_path)

    num_plots = len(columns)
    n_cols = 2
    n_rows = (num_plots + n_cols - 1) // n_cols  # Adjusted calculation for n_rows
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows), squeeze=False)  # Ensure axs is always 2D
    
    # Plotting the averaged data and standard deviation for each column
    j = 0
    for i, col in enumerate(columns):
        if col not in df_mean.columns:
            continue  # Skip columns not found in the averaged DataFrame
        
        if i % n_cols == 0 and i != 0:
            j += 1
        
        ax = axs[j, i % n_cols]
        x = df_mean.index
        y = df_mean[col]
        std = df_std[col]
        
        ax.plot(x, y, label=f'Average {col}')
        ax.fill_between(x, y-std, y+std, alpha=0.2, label='Standard Deviation')
        
        ax.legend()
        
        ax.set_title(f'Average Data for: {col}')
        ax.set_xlabel('Index')
        ax.set_ylabel(col)
    
    plt.tight_layout()
    plt.savefig(os.path.join(exp_path, 'average_training_plots_with_std_wqkl.png'))
    plt.close()


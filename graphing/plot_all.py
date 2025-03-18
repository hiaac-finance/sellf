import os

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

CMAP ={ }

"""
for plotting comparisons across different models
"""
# uses eval data
def plot_bank_cash_over_time(names, datas, save_dir, cmap=None, vertical_line_index=None, plot_std=False, is_seed_comp=False, step_lim=2000):
    if cmap is None:
        cmap = CMAP
    avg_datas = []
    for name, data in zip(names, datas):
        aggregated_tot_ep_bank_cash = data['tot_bank_cash_over_time']
        
        
        mean_val = np.mean(aggregated_tot_ep_bank_cash, axis=0)[:step_lim]
        std_val = np.std(aggregated_tot_ep_bank_cash, axis=0)[:step_lim]
        timesteps = list(range(step_lim))
        plt.plot(timesteps, mean_val, label=f'{name}', c=cmap[name])
        if plot_std:
            plt.fill_between(timesteps, mean_val - std_val, mean_val + std_val, color=cmap[name], alpha=0.2, label=f'std-deviation')

    if vertical_line_index is not None:
        plt.axvline(x=vertical_line_index, color='red', linestyle='--')

    # plt.title(f'Average Bank Cash Over Time')
    plt.xlabel('Timestep', fontsize=15)
    plt.ylabel('Bank Cash', fontsize=15)
    # plt.ylim((9900, 12000))
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('Bank Cash', fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'tot_bank_cash_over_time.png'))
    plt.close()

    if is_seed_comp:
        return 'tot_bank_cash_over_time', np.stack(avg_datas, axis=0)

# uses eval data
def plot_g_by_group_over_time(names, datas, save_dir, cmap=None, vertical_line_index=None, plot_std=False, g_type='g_sum', step_lim=2000):
    d_name = 'tot_' + g_type
    if cmap is None:
        cmap = CMAP
    for name, data in zip(names, datas):
        tot_g_over_time_mean = np.mean(data[d_name], axis=0)[:step_lim]
        tot_g_over_time_std = np.std(data[d_name], axis=0)[:step_lim]

        timesteps = np.arange(tot_g_over_time_mean.shape[0])
        plt.plot(timesteps, tot_g_over_time_mean[:,0], c=cmap[name], label=f'G0_{name}')
        plt.plot(timesteps, tot_g_over_time_mean[:,1], c=cmap[name], label=f'G1_{name}', linestyle='dotted')
        if plot_std:
            plt.fill_between(timesteps, tot_g_over_time_mean[:,0] - tot_g_over_time_std[:,0], tot_g_over_time_mean[:,0] + tot_g_over_time_std[:,0], color=cmap[name], alpha=0.2)
            plt.fill_between(timesteps, tot_g_over_time_mean[:,1] - tot_g_over_time_std[:,1], tot_g_over_time_mean[:,1] + tot_g_over_time_std[:,1], color=cmap[name], alpha=0.4)

    if vertical_line_index is not None:
        plt.axvline(x=vertical_line_index, color='red', linestyle='--')

    if g_type == 'g_pi0_sum':
        g = "g'"
        policy = 'pi_0'
    else:
        g = "g"
        policy = 'pi'

    # plt.title('Cumulative loans')
    plt.xlabel('Timestep (t)', fontsize=15)
    plt.ylabel(f"Cumulative Qual change ({g})", fontsize=15)
    plt.title(f"Total Qualification Gain ({g}, {policy}) over time by group", fontsize=15)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'tot_{g_type}_by_group_over_time.png'))
    plt.close()

# uses eval data
def plot_g_disparity_over_time(names, datas, save_dir, cmap=None, vertical_line_index=None, plot_std=False, is_seed_comp=False, g_type='g_sum', step_lim=2000):
    d_name = 'tot_' + g_type
    if cmap is None:
        cmap = CMAP
    avg_datas = []
    for name, data in zip(names, datas):
        tot_g_over_time = data[d_name]
        avg_datas.append(np.mean(tot_g_over_time, axis=0))

        g_diff = np.abs(tot_g_over_time[...,0] - tot_g_over_time[...,1])
        
        g_diff_mean = np.mean(g_diff, axis=0)[:step_lim]
        g_diff_std = np.std(g_diff, axis=0)[:step_lim]

        timesteps = np.arange(g_diff_mean.shape[0])
        plt.plot(timesteps, g_diff_mean, c=cmap[name], label=f'{name}')
        if plot_std:
            plt.fill_between(timesteps, g_diff_mean - g_diff_std, g_diff_mean + g_diff_std, color=cmap[name], alpha=0.2)

    if vertical_line_index is not None:
        plt.axvline(x=vertical_line_index, color='red', linestyle='--')

    if g_type == 'g_pi0_sum':
        g = "g'"
        policy = 'pi_0'
    else:
        g = "g"
        policy = 'pi'

    # plt.title('Cumulative loans')
    plt.xlabel('Timestep', fontsize=15)
    plt.ylabel(f"Qualification Gain Disparity ({g})", fontsize=15)
    plt.title(f"Qualification Gain Disparity ({g}, {policy}) over time", fontsize=15)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.title('Group Qualification Gain Disparity')
    plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, f'tot_{g_type}_disparity_over_time_igcorrected.png'))
    plt.savefig(os.path.join(save_dir, f'tot_{g_type}_disparity_over_time.png'))
    plt.close()

    if is_seed_comp:
        avg_datas = np.stack(avg_datas, axis=0)
        return d_name, avg_datas

# uses eval data
def plot_cscore_loan_rate_comps(names, datas, save_dir, steps, plot_std=False):
    n_rows = len(names)
    n_cols = len(steps)

    fig = plt.figure(figsize=(8 * n_cols, 6 * n_rows))
    subfigs = fig.subfigures(1, 2, hspace=0.1, width_ratios=[1, 4])

    axs0 = subfigs[0].subplots(n_rows, 1)
    axs1 = subfigs[1].subplots(n_rows, n_cols, sharex=True, sharey=True)
    
    font_multiplier = round((n_rows + n_cols) / 2)

    subfigs[0].suptitle('Models', fontsize=15*font_multiplier)
    subfigs[1].suptitle('Loan Rates up to Specific Steps', fontsize=15*font_multiplier)


    width = 0.25

    for ax, col in zip(axs1[0], steps):
        ax.set_title(f'Step {col}', fontsize=10*font_multiplier)

    for ax in axs1[-1]:
        ax.set_xlabel('Credit Score', fontsize=10*font_multiplier)

    ylabels = ['Probability' for i in range(n_rows)]
    for ax, ylabel in zip(axs1[:,0], ylabels):
        ax.set_ylabel(ylabel, fontsize=10 * font_multiplier)

    for i, (name, tot_eval_data) in enumerate(zip(names, datas)):
        axs0[i].set_axis_off()
        axs0[i].text(0.5, 0.5, name, fontsize=13*font_multiplier, ha='center', va='center')

        tot_loan_rate = np.divide(
            tot_eval_data['tot_loans_over_time_by_cscore'],
            np.where(tot_eval_data['tot_cscore_seen_over_time'] == 0, 1, tot_eval_data['tot_cscore_seen_over_time'])
        )

        for j, step in enumerate(steps):
            x = np.array(range(tot_loan_rate.shape[-1]))
            loan_rate_mean = np.mean(tot_loan_rate, axis=0)
            loan_rate_std = np.std(tot_loan_rate, axis=0)

            axs1[i, j].bar(x-width/2, loan_rate_mean[step, 0], width, label=f'Group {1}')
            axs1[i, j].bar(x+width/2, loan_rate_mean[step, 1], width, label=f'Group {2}')

            if plot_std:
                axs1[i, j].errorbar(x-width/2, loan_rate_mean[step, 0], yerr=loan_rate_std[step, 0], fmt='o', color='black')
                axs1[i, j].errorbar(x+width/2, loan_rate_mean[step, 1], yerr=loan_rate_std[step, 1], fmt='o', color='black')
            
            axs1[i,j].set_ylim(0, 1)
            axs1[i,j].tick_params(axis='both', which='major', labelsize=7*font_multiplier)

    # Add a legend for the whole figure
    handles = [
        plt.Line2D([0], [0], color='tab:blue', lw=4, label='Group 0'),
        plt.Line2D([0], [0], color='tab:orange', lw=4, label='Group 1')
    ]
    subfigs[1].legend(handles=handles, loc='upper right', fontsize=12*font_multiplier)

    # Adjust layout and save the figure
    plt.savefig(os.path.join(save_dir, 'cscore_bar_loan_rate_by_steps.png'))
    plt.close()

# uses training progress data
def plot_multi_comp(columns, df_comp_list, png_name='average_training_plots_with_std.png', save_path=None, cmap=None, plot_std=True):
    num_plots = len(columns)
    n_cols = 2
    n_rows = (num_plots + n_cols - 1) // n_cols  # Adjusted calculation for n_rows
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows), squeeze=False)  # Ensure axs is always 2D
    
    # Plotting the averaged data and standard deviation for each column
    j = 0
    for i, col in enumerate(columns):
        
        if i % n_cols == 0 and i != 0:
            j += 1
        
        ax = axs[j, i % n_cols]
        for (df, df_std) in df_comp_list:

            if cmap is not None:
                ax.set_prop_cycle(color=cmap[df.name])

            if col not in df.columns:
                continue  # Skip columns not found in the averaged DataFrame
            x = df.index
            y = df[col]
            
            ax.plot(x, y, label=f'{df.name} - {col}')

            if plot_std:
                std = df_std[col]
                ax.fill_between(x, y-std, y+std, alpha=0.2, label='Standard Deviation')
            
        ax.legend()
        ax.set_title(f'Average Data for: {col}')
        ax.set_xlabel('Index')
        ax.set_ylabel(col)
    
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        os.makedirs(save_path, exist_ok=True)
        print(f"Saving plot to {save_path}")
        plt.savefig(os.path.join(save_path, png_name))
        plt.close()
        
# uses training progress data
def plot_multi_seed_progress_data_comp(exp_paths, columns, save_path=None, seeds=None, sep_rollout_train=False, cmap=None, return_data=False):
    '''
    Plots the average and standard deviation of the specified columns for each experiment path.

    exp_paths: List of paths to the experiment directories
                should be of the form ['results_dir1/m_name1/models', 'results_dir2/m_name2/models', ...]

    columns: List of column names to plot from the CSV files

    save_path: Path to save the plot to, if None, the plot will not be saved

    seeds: List of seeds to consider for each experiment path
            if None, all seeds in the experiment path will be considered (collected w/ os.listdir())
    
    cmap: Color map to use for the plots, keys need to match the model names in experiment paths
    '''
    df_comp_list = []
    for exp_path in exp_paths:
        dfs = []  # List to store data from each seed
        
        if seeds is None:
            seed_l = [f.split('_')[-1] for f in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, f))]
        else:
            seed_l = seeds
        # Loop over each seed to load the respective CSV file
        for seed in seed_l:
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
            print(f"No data files found or specified columns missing in all files for {exp_path}.")
            return
        
        # Concatenate all dataframes along the index
        df_concat = pd.concat(dfs, axis=0)
        
        # Calculate the mean and standard deviation for each column
        df_mean = df_concat.groupby(df_concat.index).mean()
        df_std = df_concat.groupby(df_concat.index).std()

        # Save the averaged and standard deviation data to CSV files
        mean_csv_path = os.path.join(exp_path, 'average_data.csv')
        std_csv_path = os.path.join(exp_path, 'std_deviation_data.csv')

        name = exp_path.split('/')[-2]

        df_mean.name = name
        df_comp_list.append((df_mean, df_std))

    if sep_rollout_train:
        colls_roll = [col for col in columns if col.startswith('r')]
        colls_rest = [col for col in columns if not col.startswith('r')]
        plot_multi_comp(colls_roll, df_comp_list, 'avg_rollout_train_plots_wth_std.png', save_path)
        plot_multi_comp(colls_rest, df_comp_list, 'avg_train_only_plots_wth_std.png', save_path)
    else:
        plot_multi_comp(columns, df_comp_list, 'avg_train_full_plots_wth_std.png', save_path)
    
    if return_data:
        return df_comp_list
    
def smooth_columns(df, columns, window=20):
    for col in columns:
        if col in df.columns:
            # df[f'{col}_smooth'] = df[col].rolling(window=window, min_periods=0).mean()
            df[col] = df[col].rolling(window=window, min_periods=0).mean()
            # breakpoint()
    return df
    
# uses training progress data
def gather_data_frames(exp_paths, columns, seeds=None, columns_to_smooth=None):
    '''
    Compiles the average and standard deviation of the specified columns for each experiment path.

    exp_paths: List of paths to the experiment directories
                should be of the form ['results_dir1/m_name1/models', 'results_dir2/m_name2/models', ...]

    columns: List of column names to plot from the CSV files

    seeds: List of seeds to consider for each experiment path
            if None, all seeds in the experiment path will be considered (collected w/ os.listdir())
    columns_to_smooth: List of columns to apply a rolling mean to

    returns: List of tuples containing the averaged and standard deviation dataframes for each experiment path
    '''
    df_comp_list = []
    for exp_path in exp_paths:
        dfs = []  # List to store data from each seed
        
        if seeds is None:
            seed_l = [f.split('_')[-1] for f in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, f))]
        else:
            seed_l = seeds
        # Loop over each seed to load the respective CSV file
        # print(seeds)
        for seed in seed_l:
            # Construct the path to the CSV file for the current seed
            seed_path = os.path.join(exp_path, f'seed_{seed}', 'progress.csv')
            # Load the CSV file if it exists
            if os.path.exists(seed_path):
                df = pd.read_csv(seed_path)

                if columns_to_smooth:
                    df = smooth_columns(df, columns_to_smooth)
                
                # Keep only the specified columns if they exist in this file
                available_columns = set(columns).intersection(set(df.columns))
                if available_columns:
                    dfs.append(df[list(available_columns)])
        
        # Check if we successfully loaded any data
        if not dfs:
            print(f"No data files found or specified columns missing in all files for {exp_path}.")
            return
        
        # Concatenate all dataframes along the index
        df_concat = pd.concat(dfs, axis=0)
        
        # Calculate the mean and standard deviation for each column
        df_mean = df_concat.groupby(df_concat.index).mean()
        df_std = df_concat.groupby(df_concat.index).std()

        # Save the averaged and standard deviation data to CSV files
        mean_csv_path = os.path.join(exp_path, 'average_data.csv')
        std_csv_path = os.path.join(exp_path, 'std_deviation_data.csv')

        name = exp_path.split('/')[-2]

        df_mean.name = name
        df_comp_list.append((df_mean, df_std))
    
    return df_comp_list
    

def plot_seed_progress_data_comp(exp_paths, columns, m_names, save_path=None, sep_rollout_train=False, cmap=None, return_data=False):
    '''
    Plots the average and standard deviation of the specified columns for each experiment path.

    exp_paths: List of paths to the experiment directories
                should be of the form ['results_dir1/m_name1/models', 'results_dir2/m_name2/models', ...]

    columns: List of column names to plot from the CSV files

    save_path: Path to save the plot to, if None, the plot will not be saved

    seeds: List of seeds to consider for each experiment path
            if None, all seeds in the experiment path will be considered (collected w/ os.listdir())
    
    cmap: Color map to use for the plots, keys need to match the model names in experiment paths
    '''
    df_comp_list = []
    for exp_path, m_name in zip(exp_paths, m_names):
        
        # Loop over each seed to load the respective CSV file
        # Construct the path to the CSV file for the current seed
        seed_path = os.path.join(exp_path, 'progress.csv')

        # Load the CSV file if it exists
        if os.path.exists(seed_path):
            df = pd.read_csv(seed_path)
            
            # Keep only the specified columns if they exist in this file
            available_columns = set(columns).intersection(set(df.columns))
        
        # Calculate the mean and standard deviation for each column
        df_mean = df
        df_std = df

        # Save the averaged and standard deviation data to CSV files
        mean_csv_path = os.path.join(exp_path, 'average_data.csv')
        std_csv_path = os.path.join(exp_path, 'std_deviation_data.csv')
        
        # df_mean.to_csv(mean_csv_path)
        # df_std.to_csv(std_csv_path)
        
        columns = list(columns)

        df_mean.name = m_name
        df_comp_list.append((df_mean, df_std))

    if sep_rollout_train:
        colls_roll = [col for col in columns if col.startswith('r')]
        colls_rest = [col for col in columns if not col.startswith('r')]
        plot_multi_comp(colls_roll, df_comp_list, 'rollout_train_plots.png', save_path, plot_std=False)
        plot_multi_comp(colls_rest, df_comp_list, 'train_only_plots.png', save_path, plot_std=False)
    else:
        plot_multi_comp(columns, df_comp_list, 'train_full_plots.png', save_path, plot_std=False)
    
    if return_data:
        return df_comp_list


def plot_lambda_dpe(df_list, png_name='lambda_dpe_plot.png', save_path=None, cmap=None, title=None, plot_std=True, use_legend=True):    
    # Plotting the averaged data and standard deviation for each column
    column1 = 'train/rolling_lambda_loss'
    column2 = 'train/rolling_dpe_loss'
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    color_pairs = [('tab:blue', 'tab:purple'), ('tab:red', 'tab:orange')]
    
    for i, (df, df_std) in enumerate(df_list):
        x = df.index
        y1 = df[column1]
        y2 = df[column2]
        std1 = df_std[column1]
        std2 = df_std[column2]

        color1 = color_pairs[i][0]
        color2 = color_pairs[i][1]

        ax1.plot(x, y1, c=color1, label=f'$\Lambda$ Loss - {df.name}')
        if plot_std:
            ax1.fill_between(x, y1-std1, y1+std1, color=color1, alpha=0.2)

        ax2.plot(x, y2, c=color2, label=f'DPE - {df.name}', linestyle='--')
        if plot_std:
            ax2.fill_between(x, y2-std2, y2+std2, color=color2, alpha=0.2)

    # Align the zero point of the y-axes
    def align_zero_y_axes(ax1, ax2):
        # Get the current y-limits for both axes
        y1_min, y1_max = ax1.get_ylim()
        y2_min, y2_max = ax2.get_ylim()

        # Determine the maximum absolute value for each axis
        y1_abs_max = max(abs(y1_min), abs(y1_max))
        y2_abs_max = max(abs(y2_min), abs(y2_max))

        # Find the maximum absolute value between both axes
        common_max = max(y1_abs_max, y2_abs_max)

        # Set symmetric y-limits around zero for both axes
        ax1.set_ylim(-common_max, common_max)
        ax2.set_ylim(-common_max, common_max)

    align_zero_y_axes(ax1, ax2)

    ax1.set_ylabel(r'$\Lambda$ Loss', fontsize=15)
    ax2.set_ylabel(f'DPE', fontsize=15)
    ax1.set_xlabel('Number of Training Episodes', fontsize=15)
    if use_legend:
        ax1.legend(loc="upper left", fontsize=12)
        ax2.legend(loc="upper right", fontsize=12)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        os.makedirs(save_path, exist_ok=True)
        print(f"Saving plot to {save_path}")
        plt.savefig(os.path.join(save_path, png_name))
        plt.close()

import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter

"""
plotting for figures in paper
"""

def plot_multi_bank_profit_over_time(title_list, names, datas_list, save_dir, cmap=None, vertical_line_index=None, png_name=None, plot_std=False, is_seed_comp=False, lim_2k=False):
    """
    for plotting figure 3 from paper
    """
    
    n_rows = 1
    n_cols = len(datas_list)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows), sharey=True)
    # fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))

    # Convert axes to a list if there is only one subplot (single column)
    if n_cols == 1:
        axes = [axes]

    # Store handles and labels for creating a single legend
    handles = []
    labels = []

    for ax, title, datas in zip(axes, title_list, datas_list):
        for name, data in zip(names, datas):
            # Calculate mean and standard deviation
            aggregated_tot_ep_bank_cash = data['tot_bank_cash_over_time'] - 10_000
            mean_val = np.mean(aggregated_tot_ep_bank_cash, axis=0)[:500]
            std_val = np.std(aggregated_tot_ep_bank_cash, axis=0)[:500]
            timesteps = list(range(mean_val.shape[0]))

            # Plot the data
            line, = ax.plot(timesteps, mean_val, label=f'{name}', c=cmap[name], linewidth=2)
            if plot_std:
                ax.fill_between(timesteps, mean_val - std_val, mean_val + std_val, color=cmap[name], alpha=0.2, label=f'{name} std-deviation')

            # Collect the handle and label for the legend (only once per name)
            if name not in labels:
                handles.append(line)
                labels.append(name)

        # Plot a vertical line if specified
        if vertical_line_index is not None:
            ax.axvline(x=vertical_line_index, color='red', linestyle='--')

        ax.set_title(f"{title}", fontsize=26)
        ax.set_xlabel('Timestep', fontsize=24)
        ax.tick_params(axis='both', labelsize=20)
        ax.set_xlim(0, 500)

    # Set y-axis label for the first subplot only
    axes[0].set_ylabel("Bank Profit", fontsize=24)

    # Create a single legend below all subplots
    fig.legend(handles, labels, loc='lower center', fontsize=24, ncol=len(labels), bbox_to_anchor=(0.5, -0.15))

    # Adjust layout to make space for the legend below the plots
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the plot to the specified directory
    if png_name is None:
        png_name = 'multi_bank_profit_over_time.png'
    plt.savefig(os.path.join(save_dir, png_name), bbox_inches='tight')
    plt.close()


def plot_multi_g_disparity_over_time(title_list, names, datas_list, save_dir, cmap=None, vertical_line_index=None, png_name=None, plot_std=False, is_seed_comp=False, g_type='g_sum', lim_2k=False):
    """
    for plotting figure 4 from paper
    """
    d_name = 'tot_' + g_type

    n_rows = 1
    n_cols = len(datas_list)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows), sharey=True)

    # Convert axes to a list if there is only one subplot (single column)
    if n_cols == 1:
        axes = [axes]

    # Store handles and labels for creating a single legend below all subplots
    handles = []
    labels = []

    for ax, title, datas in zip(axes, title_list, datas_list):
        for name, data in zip(names, datas):
            # Calculate mean and standard deviation
            tot_g_over_time = data[d_name]
            g_diff = np.abs(tot_g_over_time[..., 0] - tot_g_over_time[..., 1])
            g_diff_mean = np.mean(g_diff, axis=0)[:500]
            g_diff_std = np.std(g_diff, axis=0)[:500]
            timesteps = np.arange(g_diff_mean.shape[0])

            # Plot the data
            line, = ax.plot(timesteps, g_diff_mean, c=cmap[name], label=f'{name}', linewidth=2)
            if plot_std:
                ax.fill_between(timesteps, g_diff_mean - g_diff_std, g_diff_mean + g_diff_std, color=cmap[name], alpha=0.2)

            # Collect the handle and label for the legend (only once per name)
            if name not in labels:
                handles.append(line)
                labels.append(name)

        # Plot a vertical line if specified
        if vertical_line_index is not None:
            ax.axvline(x=vertical_line_index, color='red', linestyle='--')

        ax.set_title(f"{title}", fontsize=26)
        ax.set_xlabel('Timestep', fontsize=24)
        ax.tick_params(axis='both', labelsize=20)
        ax.set_xlim(0, 500)

    # Set y-axis label for the first subplot only
    axes[0].set_ylabel(f"Qualification Gain Disparity", fontsize=24)

    # Create a single legend below all subplots
    fig.legend(handles, labels, loc='lower center', fontsize=24, ncol=len(labels), bbox_to_anchor=(0.5, -0.15))

    # Adjust layout to make space for the legend below the plots
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the plot to the specified directory
    if png_name is None:
        png_name = f'multi_{g_type}_disp_over_time.png'
    plt.savefig(os.path.join(save_dir, png_name), bbox_inches='tight')
    plt.close()


def plot_multi_lambda_dpe(titles, df_lists, png_name='lambda_dpe_plot.png', save_path=None, cmap=None, title=None, plot_std=True, use_legend=True):
    """
    for plotting figure 6 from paper
    """
    # df_lists is expected to be a list of tuples, where each tuple contains two df_list elements
    n_cols = len(df_lists)
    fig, axes = plt.subplots(1, n_cols, figsize=(8 * n_cols, 7.2))

    if n_cols == 1:
        axes = [axes]

    color_pairs = [('tab:blue', 'tab:purple'), ('tab:red', 'tab:orange')]

    # Store handles and labels for legend outside of the loop
    handles = []
    labels = []
    all_y1_vals = []
    all_y2_vals = []
    axes_pairs = []  # will store (ax1, ax2) references

    for idx, (title, ax, df_list) in enumerate(zip(titles, axes, df_lists)):
        ax1 = ax
        ax2 = ax1.twinx()
        axes_pairs.append((ax1, ax2))
        for i, (df, df_std) in enumerate(df_list):
            x = df.index
            y1 = df['train/rolling_lambda_loss']
            y2 = df['train/rolling_dpe_loss'].abs()
            std1 = df_std['train/rolling_lambda_loss']
            std2 = df_std['train/rolling_dpe_loss'].abs()

            color1 = color_pairs[i][0]
            color2 = color_pairs[i][1]

            all_y1_vals.extend((y1 + std1).tolist()[1:])
            all_y2_vals.extend((y2 + std2).tolist()[1:])
            all_y1_vals.extend((y1 - std1).tolist()[1:])
            all_y2_vals.extend((y2 - std2).tolist()[1:])

            line1, = ax1.plot(x, y1, c=color1, label=f'$\Lambda$ - {df.name}', linewidth=2)
            if plot_std:
                ax1.fill_between(x, y1-std1, y1+std1, color=color1, alpha=0.2)

            line2, = ax2.plot(x, y2, c=color2, label=f'DPE - {df.name}', linestyle='--', linewidth=2)
            if plot_std:
                ax2.fill_between(x, y2-std2, y2+std2, color=color2, alpha=0.2)

            # Collect handles and labels for the legend, but only once
            if idx == 0:
                handles.append(line1)
                handles.append(line2)
                labels.append(f'$\Lambda$ - {df.name}')
                labels.append(f'DPE - {df.name}')

        # Add light grey line through 0 on the y-axis for ax1
        ax1.axhline(0, color='black', linewidth=2, linestyle='--')

        # Set titles and labels for the subplots
        ax1.set_title(title, fontsize=24)

        if idx == 0:
            ax1.set_ylabel(r'Benefit Fairness ($\Lambda$)', fontsize=24)
            ax1.tick_params(axis='both', labelsize=20)
        else:
            ax1.set_yticklabels([])  # Hide y-ticks for subsequent plots
            ax1.tick_params(axis='x', labelsize=20)

        if idx == n_cols - 1:
            ax2.set_ylabel(f'DPE', fontsize=24)
            ax1.tick_params(axis='x', labelsize=20)
            ax2.tick_params(axis='y', labelsize=20)
        else:
            ax2.set_yticklabels([])  # Hide y-ticks for subsequent plots

        ax1.set_xlabel('Number of Training Episodes', fontsize=24)

    # Compute global min/max for ax1 and ax2
    global_y1_min, global_y1_max = min(all_y1_vals), max(all_y1_vals)
    global_y2_min, global_y2_max = min(all_y2_vals), max(all_y2_vals)
    # breakpoint()
    # Apply those limits to each subplot
    for (ax1, ax2) in axes_pairs:
        ax1.set_ylim(global_y1_min, global_y1_max)
        ax2.set_ylim(global_y2_min, global_y2_max)
        ax1.set_xlim(0, 1000)
        ax2.set_xlim(0, 1000)

    # Create a single legend for all subplots
    if use_legend and len(handles) > 0:
        fig.legend(handles, labels, loc='lower center', fontsize=24, ncol=len(handles)//2, bbox_to_anchor=(0.5, -0.021))

    plt.tight_layout(rect=[0, 0, 1, 0.85])  # Adjust layout to accommodate the legend
    fig.subplots_adjust(left=0.053, right=0.956, top=0.94, bottom=0.28)  # Adjust layout to make space for legend

    current_width, current_height = fig.get_size_inches()
    new_width = current_width * 0.8  # Scale width to reduce extra space
    fig.set_size_inches(new_width, current_height)

    if save_path is None:
        plt.show()
    else:
        os.makedirs(save_path, exist_ok=True)
        print(f"Saving plot to {save_path}")
        plt.savefig(os.path.join(save_path, png_name))
        plt.close()


def plot_multi_cpi_decomp(titles, df_lists, save_path, png_name=None, cmap=None, xlabel=None, ylabel=None, title=None, plot_std=True):
    """
    for plotting figure 5 from paper
    """
    n_cols = len(df_lists)
    fig, axes = plt.subplots(1, n_cols, figsize=(8 * n_cols, 7), sharey=True)

    if n_cols == 1:
        axes = [axes]

    # Store handles and labels to create a single legend below all subplots
    handles = []
    labels = []

    for idx, (ax, df_tuple) in enumerate(zip(axes, df_lists)):
        df = df_tuple[0]
        df_std = df_tuple[1]

        col_de = 'train/rolling_soft_de'
        col_ie = 'train/rolling_soft_ie'
        col_se = 'train/rolling_soft_se'
        col_cpi = 'train/rolling_c_pi_theta'

        x = df.index
        
        # Plot each line and collect handles/labels
        line1, = ax.plot(x, df[col_de].abs(), c='tab:blue', label='DPE', linewidth=2)
        line2, = ax.plot(x, df[col_ie], c='tab:orange', label='IPE', linewidth=2)
        line3, = ax.plot(x, df[col_se], c='tab:purple', label='SPE', linewidth=2)
        line4, = ax.plot(x, df[col_cpi], c='tab:green', label=r'$C_{\pi}(\theta)$', linewidth=2)
        
        if plot_std:
            ax.fill_between(x, df[col_de] - df_std[col_de], df[col_de] + df_std[col_de], color='tab:blue', alpha=0.2)
            ax.fill_between(x, df[col_ie] - df_std[col_ie], df[col_ie] + df_std[col_ie], color='tab:orange', alpha=0.2)
            ax.fill_between(x, df[col_se] - df_std[col_se], df[col_se] + df_std[col_se], color='tab:purple', alpha=0.2)
            ax.fill_between(x, df[col_cpi] - df_std[col_cpi], df[col_cpi] + df_std[col_cpi], color='tab:green', alpha=0.2)

        # Collect handles and labels only once
        if idx == 0:
            handles.extend([line1, line2, line3, line4])
            labels.extend(['DPE', 'IPE', 'SPE', r'$C_{\pi}(\theta)$'])

        ax.set_xlabel('Number of Training Episodes', fontsize=28)
        ax.set_title(titles[idx], fontsize=28)
        if idx == 0:
            ax.set_ylabel('Episode Rate', fontsize=28)

        ax.tick_params(axis='both', labelsize=24)
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
        ax.set_xlim(0, df.shape[0])

    # Create a single legend below all subplots
    fig.legend(handles, labels, loc='lower center', fontsize=28, ncol=len(handles),
               bbox_to_anchor=(0.5, -0.15))  # Position the legend below the plots

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for legend

    if save_path is None:
        plt.show()
    else:
        os.makedirs(save_path, exist_ok=True)
        print(f"Saving plot to {save_path}")
        if png_name is None:
            png_name = 'c_pi_decomp_plot.png'
        plt.savefig(os.path.join(save_path, png_name), bbox_inches='tight')
        plt.close()


def plot_cscore_loan_rate_comps_multi(names, datas, save_dir, step, png_name=None, plot_std=False):
    """
    model name should end in with beta_kl value
    """
    n_rows = 1
    n_cols = len(names)
    if step is None:
        step = 500

    # fig = plt.figure(figsize=(8 * n_cols, 6 * n_rows))
    fig, axes = plt.subplots(1, n_cols, figsize=(8 * n_cols, 6 * n_rows), sharey=True)
    # subfigs = fig.subfigures(1, 2, hspace=0.1, width_ratios=[1, 4])
    
    font_multiplier = round((n_rows + n_cols) / 2)

    width = 0.25

    for i, (name, tot_eval_data, ax) in enumerate(zip(names, datas, axes)):
    
        tot_loan_rate = np.divide(
            tot_eval_data['tot_loans_over_time_by_cscore'],
            np.where(tot_eval_data['tot_cscore_seen_over_time'] == 0, 1, tot_eval_data['tot_cscore_seen_over_time'])
        )

        # x = np.array(range(7))
        x = np.array(range(tot_loan_rate.shape[-1]))
        loan_rate_mean = np.mean(tot_loan_rate, axis=0)
        loan_rate_std = np.std(tot_loan_rate, axis=0)

        ax.bar(x-width/2, loan_rate_mean[step, 0], width, label=r'Group $s^{+}$')
        ax.bar(x+width/2, loan_rate_mean[step, 1], width, label=r'Group $s^{-}$')

        if plot_std:
            ax.errorbar(x-width/2, loan_rate_mean[step, 0], yerr=loan_rate_std[step, 0], fmt='o', color='black')
            ax.errorbar(x+width/2, loan_rate_mean[step, 1], yerr=loan_rate_std[step, 1], fmt='o', color='black')
        
        ax.set_ylim(0, 1)
        ax.tick_params(axis='both', which='major', labelsize=8*font_multiplier)
        ax.set_xlabel('Credit Score', fontsize=10*font_multiplier)

        if i == 0:
            ax.set_ylabel('Loan Rate', fontsize=10*font_multiplier)

        bkl = name.split('-')[-1]
        ax.set_title(r'$\beta ^{\Lambda}$' + f' = {bkl}', fontsize=10*font_multiplier)
        ax.legend(fontsize=10*font_multiplier)

    fig.suptitle(f'Loan Rate Comparison at Step {step}', fontsize=10*font_multiplier)

    # Adjust layout and save the figure
    if png_name is None:
        png_name = f'loan_rates_at_step_{step}.png'
    plt.savefig(os.path.join(save_dir, png_name))
    plt.close()


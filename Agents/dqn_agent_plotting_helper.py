from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import cm

import SharedConsts as SC


def create_raxml_diff_agg_plot(target_dir, episode_list, train_diff_mean, test_diff_mean,
                                    train_diff_best, test_diff_best, is_normalized=True):

    plt.clf()
    if is_normalized:
        plot_name = 'raxml-diff.png'
    else:
        plot_name = 'raxml-diff - real scale.png'
    plt.title('Agent distance from Raxml')
    plt.xlabel('# Episodes')
    plt.ylabel('Diff from Raxml')
    plt.plot(episode_list, train_diff_mean, color='m', label='train-set-mean')
    plt.plot(episode_list, test_diff_mean, color='g', label='test-set-mean')
    plt.plot(episode_list, train_diff_best, color='b', label='train-set-best')
    plt.plot(episode_list, test_diff_best, color='c', label='test-set-best')
    plt.legend()
    plt.savefig(target_dir / plot_name)
    plt.clf()


def create_improvement_percent_from_starting(target_dir, episode_list, nj_improvement_from_raxml_train,
                                             nj_improvement_from_raxml_test, our_improvement_from_raxml_train,
                                             our_improvement_from_raxml_test):

    plt.clf()
    plot_name = 'improvement_from_raxml.png'
    plt.title('Improvement from starting tree compared to Raxml')
    plt.xlabel('# Episodes')
    plt.ylabel('% of improvement from starting tree')
    plt.plot(episode_list, nj_improvement_from_raxml_train, color='m', label='train-set-nj')
    plt.plot(episode_list, nj_improvement_from_raxml_test, color='g', label='test-set-nj')
    plt.plot(episode_list, our_improvement_from_raxml_train, color='b', label='train-set-agent')
    plt.plot(episode_list, our_improvement_from_raxml_test, color='c', label='test-set-agent')
    plt.legend()
    plt.savefig(target_dir / plot_name)
    plt.clf()


def create_improvement_percent_from_nj(target_dir, episode_list, raxml_improvement_from_nj_train,
                                       raxml_improvement_from_nj_test, our_improvement_from_nj_train,
                                       our_improvement_from_nj_test):

    plt.clf()
    plot_name = 'improvement_from_nj.png'
    plt.title('Improvement from starting tree compared to nj')
    plt.xlabel('# Episodes')
    plt.ylabel('% of improvement from starting tree compared to NJ')
    plt.plot(episode_list, raxml_improvement_from_nj_train, color='m', label='train-set-raxml')
    plt.plot(episode_list, raxml_improvement_from_nj_test, color='g', label='test-set-raxml')
    plt.plot(episode_list, our_improvement_from_nj_train, color='b', label='train-set-agent')
    plt.plot(episode_list, our_improvement_from_nj_test, color='c', label='test-set-agent')
    plt.legend()
    plt.savefig(target_dir / plot_name)
    plt.clf()


def create_raxml_diff_per_data_set_plot(target_dir, episode_list, bests, means, data_set, is_normalized=True):

    plt.clf()
    if is_normalized:
        plot_name = f'raxml-diff_{data_set}.png'
    else:
        plot_name = f'raxml-diff_{data_set} - real scale.png'
    plt.title(f'Agent distance from Raxml - ll - {data_set}')
    plt.xlabel('# Episodes')
    plt.ylabel('Diff from Raxml')
    plt.plot(episode_list, bests, color='g', label='best')
    plt.plot(episode_list, means, color='m', label='mean')
    plt.legend()
    plt.savefig(target_dir / plot_name)
    plt.clf()


def create_ramxl_diff_all_points_plot(raw_results_df, target_dir, is_normalized=True):

    plt.clf()
    if is_normalized:
        res_column = 'll_diff'
        plot_name = 'raxml-diff - all points.png'
    else:
        res_column = 'll_diff_real_scale'
        plot_name = 'raxml-diff - all points - at scale.png'
    plt.title('All points distance from Raxml - across all datasets')
    plt.xlabel('run number')
    plt.ylabel('Diff from Raxml')
    p = 0
    colors = iter(cm.rainbow(np.linspace(0, 1, len(raw_results_df['data_set_number'].unique()) + 4)))
    for data_set in raw_results_df['data_set_number'].unique():
        raw_res_tmp = list(raw_results_df[raw_results_df['data_set_number'] == data_set][res_column])
        plt.scatter(range(p, p + len(raw_res_tmp)), raw_res_tmp, color=next(colors), label=str(data_set))
        p += len(raw_res_tmp)
    mean_maxs = np.mean(raw_results_df.groupby('data_set_number').max()[res_column].values)
    mean_means = np.mean(raw_results_df.groupby('data_set_number').mean()[res_column].values)
    plt.plot([o for o in range(p)], [mean_means for o in range(p)], color=next(colors),
             label=f'mean - {round(mean_means, 4)}')
    next(colors)
    next(colors)
    plt.plot([o for o in range(p)], [mean_maxs for o in range(p)], color=next(colors),
             label=f'max - {round(mean_maxs, 4)}')
    plt.legend(prop={'size': 6})
    plt.savefig(target_dir / plot_name)
    plt.clf()


def plot_loss(result_dir, epoch_loss):

    plt.clf()
    plt.xlabel('# Epochs')
    plt.plot(epoch_loss, color='m', label=f'last epoch loss={epoch_loss[-1]:.3f}')
    plt.title("DQN loss")
    plt.legend()
    plt.savefig(result_dir / 'DQN-Loss.png', bbox_inches='tight')
    plt.clf()


def make_violin_plot(result_dir, diff_lst):

    plt.clf()
    sns.violinplot(y=diff_lst)
    plt.ylabel('difference between RL best ll and ramxl best ll')
    plt.title('our best ll vs. raml best ll')
    plt.savefig(result_dir / 'test_violin_plot.png')
    plt.clf()


def plot_specific_agent_run_all_moves(results, hill_climb_results,  travel_log_index,
                                      raxml_best_ll, date_set, target_dir):

    plt.clf()
    x_scale = [i for i in range(len(results))]
    plt.plot(x_scale, results, color='#1f77b4', label='RL episodes ll')
    plt.plot(x_scale, [raxml_best_ll for i in range(len(results))], color='#2ca02c', label='Raxml best ll benchmark')

    plt.plot(x_scale, hill_climb_results, color='#ff7f0e', label='Greedy ll')
    plt.xticks([i for i in range(0, len(results), 2)])

    plt.legend()
    plt.xlabel('Move number')
    plt.ylabel('Likelihood score')
    # plt.title(f'Test episodes ll\'s in dataset {date_set}')
    fig_name = f"{date_set}_full_travel_log_{travel_log_index}" + ".png"
    plt.savefig(str(target_dir / fig_name))
    plt.clf()


def plot_specific_agent_run_all_moves_rf(agent_reached_trees, ml_tree_obj, date_set, target_dir):

    plt.clf()
    x_scale = [i for i in range(len(agent_reached_trees))]
    plt.plot(x_scale, [ml_tree_obj.robinson_foulds(t, unrooted_trees=True)[0] for t in agent_reached_trees],
             color='blue', label='RL episodes rf')
    plt.xlabel('Move number')
    plt.ylabel('Current trees rf from raxml ml tree')
    plt.title(f'Test episodes rf\'s in dataset {date_set}')
    plt.ylim(0.0, plt.ylim()[1])
    plt.savefig(target_dir / (date_set + '_rf.png'))
    plt.clf()


def create_variance_plot_per_dataset(experiment_folder_name, data_set_to_plot=None):
    base_folder = SC.EXPERIMENTS_RESDIR / experiment_folder_name / SC.RESULTS_FOLDER_NAME
    variance_plots_folder = base_folder / 'per_data_set_variance'
    variance_plots_folder.mkdir(exist_ok=True)

    # collect dfs
    dfs_with_id = []
    for file in [f for f in glob(str(base_folder) + '/**/**/*') if str(f).find('raw_results_test_set.csv') != -1]:
        df = pd.read_csv(file)
        episode_str = file.split('/')[-3].split('_')[-1]
        df['episode'] = episode_str
        df.sort_values('episode', inplace=True)
        if 'll_diff_real_scale' not in df.columns:
            df['ll_diff_real_scale'] = df['ll_diff'] * df['ll_norm_factor']
        dfs_with_id.append(df)
    all_exps_df = pd.concat(dfs_with_id)
    for data_set in all_exps_df['data_set_number'].unique():
        if data_set_to_plot is not None and str(data_set) != str(data_set_to_plot):
            continue
        data_set_data = all_exps_df[all_exps_df['data_set_number'] == data_set]
        plt.clf()
        plt.title(f'All points distance from Raxml - across all episodes - {data_set}')
        plt.xlabel('Test run')
        plt.ylabel('Diff from Raxml normalized')
        p = 0
        colors = iter(cm.rainbow(np.linspace(0, 1, len(data_set_data['episode'].unique()) + 4)))
        for episode in data_set_data['episode'].unique():
            raw_res_tmp = list(data_set_data[data_set_data['episode'] == episode]['ll_diff'])
            plt.scatter(range(p, p + len(raw_res_tmp)), raw_res_tmp, color=next(colors), label=str(episode))
            p += len(raw_res_tmp)

        plt.legend(prop={'size': 6})
        plt.savefig(variance_plots_folder / f'variance_plot_{data_set}.png')
        plt.clf()

        plt.clf()
        plt.title(f'All points distance from Raxml - across all episodes - {data_set}')
        plt.xlabel('Test run')
        plt.ylabel('Diff from Raxml')
        p = 0
        colors = iter(cm.rainbow(np.linspace(0, 1, len(data_set_data['episode'].unique()) + 4)))
        for episode in data_set_data['episode'].unique():
            raw_res_tmp = list(data_set_data[data_set_data['episode'] == episode]['ll_diff_real_scale'])
            plt.scatter(range(p, p + len(raw_res_tmp)), raw_res_tmp, color=next(colors), label=str(episode))
            p += len(raw_res_tmp)

        plt.legend(prop={'size': 6})
        plt.savefig(variance_plots_folder / f'variance_plot_{data_set}_at_scale.png')
        plt.clf()


def log_amazing_moves(amazing_moves, target_dir):
    if len(amazing_moves) == 0:
        return
    else:
        print(f'THERE WERE AMAZING MOVES! go check it out - they are at {target_dir}')
        all_amazing_moves = [j for i in amazing_moves.values() for j in i]
        df = pd.DataFrame(all_amazing_moves)
        df.to_csv(str(target_dir / 'amazing_moves.csv'))

import gc
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import cm


class RandomWalkAgent:
    """
    an agent to preform random walks and plot its performance.
    """

    def __init__(self, env, result_dir):
        """
        :param env: a dqn agent env
        :param result_dir: name of directory to save results
        """
        self.result_dir = Path(result_dir)
        self.env = env
        self.raxml_diff_log = ([], [], [], [])
        self.raxml_res_per_ds = {}

    def preform_random_walk(self, num_of_walk_per_dataset=20):
        full_run_examples_dir = self.result_dir / 'Full walk examples'
        full_run_examples_dir.mkdir(exist_ok=True)
        per_ds_graphs_dir = self.result_dir / 'Per dataset aggs'
        per_ds_graphs_dir.mkdir(exist_ok=True)

        train_mean_lst, train_mean_lst_rf = [], []
        train_testing_log, test_testing_log = {}, {}
        raxml_res_per_ds = self.raxml_res_per_ds
        raw_results_train_set_df = pd.DataFrame(columns=["test_replication_number", "data_set_number", "ll_diff", "rf_dist", "ll_norm_factor", "rf_norm_factor"])

        for i in range(num_of_walk_per_dataset):
            train_diff_lst, specific_test_diffs_norms_per_ds_train, train_rf_dists, specific_test_rf_norms_per_ds_train = self.perform_random_walk_tests(
                self.result_dir, full_run_examples_dir, verbose=(i == 0))
            train_mean_lst.append(np.mean(train_diff_lst))
            train_mean_lst_rf.append(np.mean(train_rf_dists))

            for train_data_set, raxml_diff_norm in specific_test_diffs_norms_per_ds_train.items():
                raxml_diff, norm_factor_ll = raxml_diff_norm
                rf_score, norm_factor_rf = specific_test_rf_norms_per_ds_train.get(train_data_set)
                if raxml_res_per_ds.get(train_data_set) is None:
                    raxml_res_per_ds[train_data_set] = ([], [])

                if train_testing_log.get(train_data_set) is None:
                    train_testing_log[train_data_set] = {'diffs': [],
                                                         'diffs_rf': []}

                train_testing_log[train_data_set]['max'] = max(raxml_diff,
                                                               train_testing_log.get(train_data_set).get('max',
                                                                                                         -sys.maxsize))
                train_testing_log[train_data_set]['diffs'].append(raxml_diff)

                train_testing_log[train_data_set]['min_rf'] = min(specific_test_rf_norms_per_ds_train.get(train_data_set)[0],
                                                               train_testing_log.get(train_data_set).get('min_rf',
                                                                                                         sys.maxsize))
                train_testing_log[train_data_set]['diffs_rf'].append(specific_test_rf_norms_per_ds_train.get(train_data_set)[0])
                raw_results_train_set_df = raw_results_train_set_df.append(pd.Series(dict(zip(raw_results_train_set_df.columns, [str(i), train_data_set, raxml_diff, rf_score, norm_factor_ll, norm_factor_rf]))), ignore_index=True)

        # gets updated inplace
        train_diff_mean, test_diff_mean, train_diff_best, test_diff_best = self.raxml_diff_log
        train_diff_mean.append(np.mean(train_mean_lst))
        train_diff_best.append(np.mean([val['max'] for val in train_testing_log.values()]))
        raw_results_train_set_df['actual_diff'] = raw_results_train_set_df['ll_diff'] * raw_results_train_set_df['ll_norm_factor']
        raw_results_train_set_df.to_csv(self.result_dir / 'raw_results_train_set.csv')

        plt.title('Random Walk distance from Raxml')
        plt.xlabel('Random walk run')
        plt.ylabel('Diff from Raxml normalized')
        p = 0
        colors = iter(cm.rainbow(np.linspace(0, 1, len(raw_results_train_set_df['data_set_number'].unique()) + 4)))
        for data_set in raw_results_train_set_df['data_set_number'].unique():
            raw_res_tmp = list(raw_results_train_set_df[raw_results_train_set_df['data_set_number'] == data_set]['ll_diff'])
            plt.scatter(range(p, p+len(raw_res_tmp)), raw_res_tmp, color=next(colors), label=str(data_set))
            p += len(raw_res_tmp)
        mean_maxs = np.mean(raw_results_train_set_df.groupby('data_set_number').max()['ll_diff'].values)
        mean_means = np.mean(raw_results_train_set_df.groupby('data_set_number').mean()['ll_diff'].values)
        plt.plot([o for o in range(p)], [mean_means for o in range(p)], color=next(colors), label=f'mean - {round(mean_means,4)}')
        next(colors)
        next(colors)
        plt.plot([o for o in range(p)], [mean_maxs for o in range(p)], color=next(colors), label=f'max - {round(mean_maxs,4)}')
        plt.legend(prop={'size': 6})
        plt.savefig(self.result_dir / 'raxml-diff.png')
        plt.clf()

        plt.title('Random Walk distance from Raxml')
        plt.xlabel('Random walk run')
        plt.ylabel('Diff from Raxml')
        p = 0
        colors = iter(cm.rainbow(np.linspace(0, 1, len(raw_results_train_set_df['data_set_number'].unique()) + 4)))
        for data_set in raw_results_train_set_df['data_set_number'].unique():
            raw_res_tmp = list(raw_results_train_set_df[raw_results_train_set_df['data_set_number'] == data_set]['actual_diff'])
            plt.scatter(range(p, p+len(raw_res_tmp)), raw_res_tmp, color=next(colors), label=str(data_set))
            p += len(raw_res_tmp)

        mean_maxs = np.mean(raw_results_train_set_df.groupby('data_set_number').max()['actual_diff'].values)
        mean_means = np.mean(raw_results_train_set_df.groupby('data_set_number').mean()['actual_diff'].values)
        plt.plot([o for o in range(p)], [mean_means for o in range(p)], color=next(colors), label=f'mean - {round(mean_means,4)}')
        next(colors)
        next(colors)
        plt.plot([o for o in range(p)], [mean_maxs for o in range(p)], color=next(colors), label=f'max - {round(mean_maxs,4)}')
        plt.legend(prop={'size': 6})
        plt.savefig(self.result_dir / 'raxml-diff-not normalized.png')
        plt.clf()

        # create plot for each dataset

        for data_set, raxml_diffs in raxml_res_per_ds.items():
            raw_res_tmp = list(raw_results_train_set_df[raw_results_train_set_df['data_set_number'] == data_set]['ll_diff'])

            plt.title(f'Agent distance from Raxml - ll - {data_set}')
            plt.xlabel('# Episodes')
            plt.ylabel('Diff from Raxml')
            plt.scatter(range(len(raw_res_tmp)), raw_res_tmp, color='g')

            plt.savefig(per_ds_graphs_dir / f'raxml-diff_{data_set}.png')

            plt.clf()

            raw_res_tmp = list(raw_results_train_set_df[raw_results_train_set_df['data_set_number'] == data_set]['actual_diff'])

            plt.title(f'Agent distance from Raxml - ll - not normalized - {data_set}')
            plt.xlabel('# Episodes')
            plt.ylabel('Diff from Raxml')
            plt.scatter(range(len(raw_res_tmp)), raw_res_tmp, color='g')

            plt.savefig(per_ds_graphs_dir / f'raxml-diff_{data_set}_not_normalized.png')

            plt.clf()

        self.raxml_res_per_ds = raxml_res_per_ds

        not_normalized_res = np.mean(raw_results_train_set_df.groupby('data_set_number').max()['actual_diff'].values)
        return train_diff_best[-1], not_normalized_res

    @staticmethod
    def make_violin_plot(result_dir, diff_lst):
        sns.violinplot(y=diff_lst)
        plt.ylabel('difference between RL best ll and ramxl best ll')
        plt.title('our best ll vs. raml best ll')
        plt.savefig(result_dir / 'test_violin_plot.png')
        plt.clf()

    @staticmethod
    def log_amazing_moves(main_results_dir, likelihood_from_env, raxml_best_ll, full_path):
        try:
            with open(main_results_dir / 'good_results_log.txt', 'a') as fp:
                fp.write("Our agent visited a tree with *higher* likelihood than RaxML's !\n "
                         "{} compared to {} in dataset:\n{}.\n".format(likelihood_from_env, raxml_best_ll, full_path))
        except:
            print("failed writing details to a file 'good_results_log.txt'.\n"
                  "In any case, our agent visited a tree with *higher* likelihood than RaxML's !")

    def perform_random_walk_tests(self, main_results_dir, test_result_dir, verbose=False):
        """
        tests agent on all datasets in test_env, test_env is just a var name- it's ok to send the train env
        """
        # save the old env
        agent_travel_log = {}  # string to list dict
        max_likelihood_reached, diffs_norms_per_ds, rf_norms_per_ds = {}, {}, {}
        diff_lst, rf_diff_lst = [], []

        for dataset in self.env.helper.all_data_sets:
            gc.collect()
            self.env.reset(dataset=dataset)
            agent_travel_log[dataset] = [[self.env.likelihood], [self.env.tree]]
            raxml_best_ll = self.env.get_raxml_likelihood()
            done = False

            while not done:
                actions = self.env.get_all_possible_actions()

                # act and get feedback from env
                reward, done = self.env.step(random.choice(range(len(actions))))
                likelihood_from_env = self.env.likelihood
                agent_travel_log[dataset][0].append(likelihood_from_env)
                agent_travel_log[dataset][1].append(self.env.tree)

                if likelihood_from_env > raxml_best_ll:
                    self.log_amazing_moves(main_results_dir, str(likelihood_from_env), str(raxml_best_ll), test_result_dir / dataset)

            # our horizon could force the agent to preform a last bad move
            max_likelihood_reached[dataset] = max(agent_travel_log[dataset][0][-1], agent_travel_log[dataset][0][-2])
            specific_norm_factor_ll = self.env.helper.get_normalization_factor_per_ds(data_set=dataset) # normalization factor for dll (NJ tree -likelihood)
            specific_norm_factor_ll *= -1 # for te reward, this is done independently. the factor is always a negative value. consider changing the the factor itself to be the abs value
            raxml_diff = (max_likelihood_reached[dataset] - raxml_best_ll)
            raxml_diff /= specific_norm_factor_ll
            diff_lst.append(raxml_diff)
            diffs_norms_per_ds[dataset] = (raxml_diff, specific_norm_factor_ll)

            specific_norm_factor_rf, rf_distance = 1, 1
            rf_distance /= specific_norm_factor_rf
            rf_diff_lst.append(rf_distance)
            rf_norms_per_ds[dataset] = (rf_distance, specific_norm_factor_rf)

            # plot line plot of the first replicate run (only)
            if verbose:
                x_scale = [i for i in range(len(agent_travel_log[dataset][0]))]
                plt.plot(x_scale, agent_travel_log[dataset][0], color='blue', label='RL episodes ll')
                plt.plot(x_scale, [raxml_best_ll for i in range(len(agent_travel_log[dataset][0]))], color='red',
                         label='Raxml best ll benchmark')
                plt.xlabel('Move number')
                plt.ylabel('Current trees ll')
                plt.title(f'Test episodes ll\'s in dataset {dataset}')
                plt.savefig(test_result_dir / (dataset + '.png'))
                plt.clf()

        return diff_lst, diffs_norms_per_ds, rf_diff_lst, rf_norms_per_ds


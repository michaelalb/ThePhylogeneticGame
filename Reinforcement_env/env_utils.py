import os
import random
import sys
import time
from queue import Queue
from subprocess import call, PIPE, STDOUT, Popen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import signal

import FeatureExtractor.FeatureExtractorGeneralToolBox as FGT
import SPR_generator.SPR_move as spr_move
import SharedConsts as SC
from FeatureExtractor.FeatureExtractor import FeatureExtractorClass
from SPR_generator.SPR_move import get_moves_from_obj, generate_tree_object


class PhyloGameUtils:
    """
    this class holds additional environment data
    which is not part of the main algorithm
    """

    def __init__(self, feature_extractor_instance: FeatureExtractorClass, results_dir, datasets=None,
                 use_random_starts=False):
        self.data_set = None
        self.fe = feature_extractor_instance
        self.use_random_starts = use_random_starts
        self.results_dir = results_dir
        # check if user asks to set env to specific data sets
        if datasets:
            # it's ok to be lazy and send ints
            self.all_data_sets = [str(x) for x in datasets]
        else:
            self.all_data_sets = [f.name for f in SC.PATH_TO_RAW_TREE_DATA.iterdir() if f.is_dir()]
        self.starting_trees = self.init_starting_trees()
        self.reward_normalizations = self.create_normalization_factors()

    def create_normalization_factors(self):
        reward_normalizations = {}
        for data_set in self.all_data_sets:
            nj_tree = generate_tree_object(SC.PATH_TO_RAW_TREE_DATA / data_set / SC.NJ_STARTING_TREE_FILE_NAME)
            nj_ll = self.fe.extract_likelihood(nj_tree, data_set)
            reward_normalizations[data_set] = nj_ll['current_ll'][0]
        return reward_normalizations

    def get_normalization_factor_per_ds(self, data_set):
        return self.reward_normalizations[data_set]  # didnt use get on purpose to raise error if no factor is found

    def get_horizon(self, tree_obj=None):
        if not tree_obj:
            # if tree_obj is not sent - just take tree from last data_set
            start_tree_path = SC.PATH_TO_RAW_TREE_DATA / self.data_set / SC.NJ_STARTING_TREE_FILE_NAME
            tree_obj = spr_move.generate_tree_object(start_tree_path)

        if SC.HORIZON == 'do not use':
            moves = get_moves_from_obj(tree_obj)
            return int(SC.HORIZON_MULTIPLIER * len(moves))
        else:
            return SC.HORIZON

    def get_starting_tree(self, data_set_num=None):
        if data_set_num is None:
            self.data_set = random.choice(self.all_data_sets)
        else:
            self.data_set = data_set_num
        if self.use_random_starts:
            tree_obj = self.get_random_tree(self.data_set)
        else:
            start_tree_path = SC.PATH_TO_RAW_TREE_DATA / self.data_set / SC.NJ_STARTING_TREE_FILE_NAME
            tree_obj = spr_move.generate_tree_object(start_tree_path)

        return tree_obj

    @staticmethod
    def transform(feature_lst):
        feature_lst = np.array(feature_lst).squeeze()

        if True in np.isnan(feature_lst):
            raise Exception("Error: env_utils input contains NaN!, terrible")

        dtype = SC.DTYPE
        if SC.USE_CUDA:
            feature_tensor = torch.tensor(feature_lst, dtype=dtype).cuda()
        else:
            feature_tensor = torch.tensor(feature_lst, dtype=dtype)

        return feature_tensor

    @staticmethod
    def plot_rl_scores(result_dir, rewards, start_q, loss, epsilons=None):
        fig = plt.figure()
        gs = fig.add_gridspec(2, hspace=0)
        (ax1, ax2) = gs.subplots(sharex=True)
        ax1.plot(np.arange(len(rewards)), rewards, alpha=.5)
        ax2.plot(np.arange(len(start_q)), start_q)
        ax1.set_ylabel('episode-total-reward')  # edit later to fit the desired scores to present
        ax2.set_ylabel('start-Q-value')  # edit later to fit the desired scores to present
        ax2.set_xlabel('# Episodes')
        ax1.label_outer()
        ax2.label_outer()

        window_length = 101  # window length must be less than len(rewards) and odd
        if len(rewards) > window_length:
            smoothed_reward = signal.savgol_filter(rewards, window_length=window_length, polyorder=3)
            ax1.plot(np.arange(len(rewards)), smoothed_reward, color='r')

        plt.savefig(result_dir / 'Predictions.png')
        plt.close(fig)

        plt.xlabel('# Times Learned')
        if len(loss) > 0:
            plt.plot(range(len(loss)), loss, color='m',  label=f'last DQN loss={loss[-1]:.3f}')
            plt.legend()
        else:
            plt.plot(range(len(loss)), loss, color='m')
        plt.title("DQN loss")
        plt.savefig(result_dir / 'DQN-Loss.png', bbox_inches='tight')
        plt.clf()

        if epsilons:
            plt.xlabel('# steps')
            plt.ylabel('epsilon')
            plt.plot(range(len(epsilons)), epsilons, color='orange')
            plt.ticklabel_format(useOffset=False)
            plt.title("epsilon change")
            plt.savefig(result_dir / 'Epsilons.png')
            plt.clf()

    def get_random_tree(self, data_set):
        """
        this is an auxiliary function to create a totally random starting tree with no brach length optimization
        :param data_set:
        :return:
        """
        return self._get_random_tree(data_set)

    def _get_random_tree(self, data_set):
        """
        this is an internal function designed to provide the next random start tree
        :param data_set:
        :return:
        """
        current_start_tree_queue = self.starting_trees.get(data_set)
        if current_start_tree_queue.empty():
            # refill queue
            random_trees_path = os.path.join(str(self.results_dir), 'random_starting_trees',
                                             SC.RANDOM_STARTING_TREES_FILE_NAME.format(data_set=data_set))
            parsimony_trees_path = os.path.join(str(self.results_dir), 'random_starting_trees',
                                             SC.RANDOM_STARTING_TREES_FILE_NAME.format(data_set=data_set))
            self.fill_queue_relevant_newicks(random_trees_path, parsimony_trees_path, current_start_tree_queue, data_set)
        start_tree = current_start_tree_queue.get()
        return start_tree

    def init_starting_trees(self, specific_data_set=None):
        """
        this function initializes all the random starting trees for the run and puts them in queues
        :return:
        """

        starting_trees_queue_dict = {}
        all_ds = self.all_data_sets if specific_data_set is None else specific_data_set
        for data_set in all_ds:
            self.generate_starting_file_for_data_set(data_set)
            # self.generate_starting_file_for_data_set(data_set, is_random=False)
            current_queue = Queue()
            random_trees_path = os.path.join(str(self.results_dir), 'random_starting_trees',
                                             SC.RANDOM_STARTING_TREES_FILE_NAME.format(data_set=data_set))
            parsimony_trees_path = os.path.join(str(self.results_dir), 'parsimony_starting_trees',
                                             SC.RANDOM_STARTING_TREES_FILE_NAME.format(data_set=data_set))
            self.fill_queue_relevant_newicks(random_trees_path, parsimony_trees_path, current_queue, data_set)
            starting_trees_queue_dict[data_set] = current_queue
        return starting_trees_queue_dict

    def remove_files(self):
        """
        removes all redundant files created
        :return:
        """
        all_ds = self.all_data_sets
        for data_set in all_ds:
            random_trees_path = os.path.join(str(self.results_dir), 'random_starting_trees',
                                             SC.RANDOM_STARTING_TREES_FILE_NAME.format(data_set=data_set))
            parsimony_trees_path = os.path.join(str(self.results_dir), 'parsimony_starting_trees',
                                             SC.RANDOM_STARTING_TREES_FILE_NAME.format(data_set=data_set))
            if os.path.isfile(random_trees_path):
                os.remove(random_trees_path)
            if os.path.isfile(parsimony_trees_path):
                os.remove(parsimony_trees_path)

    def fill_queue_relevant_newicks(self, random_trees_path, parsimony_trees_path, current_queue, data_set):
        finished_files = False
        if not os.path.isfile(random_trees_path):
            self.generate_starting_file_for_data_set(data_set)

        # if not os.path.isfile(parsimony_trees_path):
        #     self.generate_starting_file_for_data_set(data_set, is_random=False)

        with open(random_trees_path, 'r') as fp:
            relevant_random_newicks = [fp.readline() for x in range(SC.RANDOM_STARTING_TREES_MEMORY_BATCH_FOR_DS)]
        # with open(parsimony_trees_path, 'r') as fp:
        #     relevant_parsimony_newicks = [fp.readline() for x in range(SC.RANDOM_STARTING_TREES_MEMORY_BATCH_FOR_DS)]
        for i in range(SC.RANDOM_STARTING_TREES_MEMORY_BATCH_FOR_DS):
            relevant_random_newick = relevant_random_newicks[i]
            # relevant_parsimony_newick = relevant_parsimony_newicks[i]
            if relevant_random_newick == '': # and relevant_parsimony_newick == '':
                finished_files = True
                break
            current_queue.put(generate_tree_object(relevant_random_newick))
            # current_queue.put(generate_tree_object(relevant_parsimony_newick))
        if not finished_files:
            p = call(['sed', '-i', f'1,{SC.RANDOM_STARTING_TREES_MEMORY_BATCH_FOR_DS}d', random_trees_path])
            # p = call(['sed', '-i', f'1,{SC.RANDOM_STARTING_TREES_MEMORY_BATCH_FOR_DS}d', parsimony_trees_path])
        else:
            os.remove(random_trees_path)
            self.generate_starting_file_for_data_set(data_set)
            # self.generate_starting_file_for_data_set(data_set, is_random=False)

    def generate_starting_file_for_data_set(self, data_set, is_random=True):
        """
        creates relevant random starting trees file
        :param data_set: for which data set to do so
        :param is_random: indicates whether to generate random or parsimony starting trees
        :return:
        """
        # one for each episode, 10 for every test, and for the last test + something to offset randomness
        how_many_trees_to_generate = 1000
        folder_suffix_str = 'random_starting_trees' if is_random else 'parsimony_starting_trees'
        raxml_command = 'rand' if is_random else 'parsimony'
        path_to_starting_trees = os.path.join(str(self.results_dir), folder_suffix_str)
        if not os.path.exists(path_to_starting_trees):
            os.mkdir(path_to_starting_trees)

        path_to_msa = str(SC.PATH_TO_RAW_TREE_DATA / data_set / SC.MSA_FILE_NAME)

        stat_path = str(SC.PATH_TO_RAW_TREE_DATA / data_set / SC.PHYML_PARAM_FILE_NAME)
        freq, rates, pinv, alpha = FGT.get_likelihood_params(stat_path)
        alpha = alpha if float(alpha) > 0.02 else 0.02
        model_line_params = 'GTR{rates}+I{pinv}+G{alpha}+F{freq}'.format(rates="{{{0}}}".format("/".join(rates)),
                                                                         pinv="{{{0}}}".format(pinv),
                                                                         alpha="{{{0}}}".format(alpha),
                                                                         freq="{{{0}}}".format("/".join(freq)))

        # make sure no raxml files already exist
        for file in [os.path.join(path_to_starting_trees, f"{data_set}.raxml.rba"),
                     os.path.join(path_to_starting_trees, f"{data_set}.raxml.log"),
                     os.path.join(path_to_starting_trees, f"{data_set}.raxml.reduced.phy"),
                     os.path.join(path_to_starting_trees, SC.RANDOM_STARTING_TREES_FILE_NAME.format(data_set=data_set))]:
            if os.path.isfile(file):
                os.remove(str(file))

        # raxml-ng insists on getting a file and creating the extra files
        prefix = os.path.join(path_to_starting_trees, str(data_set))
        p = call(
            [SC.RAXML_NG_SCRIPT, '--start', '--msa', path_to_msa, '--threads', '1', '--opt-branches', 'on',
             '--opt-model', 'off', '--model', model_line_params, '--tree', raxml_command + '{'+str(how_many_trees_to_generate)+'}',
             '--prefix', prefix, '--seed', str(random.randint(0, sys.maxsize))],
            stdout=PIPE, stdin=PIPE, stderr=STDOUT)

        # delete bad files that raxml creates
        for file in [os.path.join(path_to_starting_trees, f"{data_set}.raxml.rba"),
                     os.path.join(path_to_starting_trees, f"{data_set}.raxml.log"),
                     os.path.join(path_to_starting_trees, f"{data_set}.raxml.reduced.phy")]:
            if os.path.isfile(file):
                os.remove(str(file))

    def generate_raxml_max_liklihood_tree(self, data_set, total_trees):
        path_to_msa = str(SC.PATH_TO_RAW_TREE_DATA / data_set / SC.MSA_FILE_NAME)
        stat_path = str(SC.PATH_TO_RAW_TREE_DATA / data_set / SC.PHYML_PARAM_FILE_NAME)
        freq, rates, pinv, alpha = FGT.get_likelihood_params(stat_path)
        alpha = alpha if float(alpha) > 0.02 else 0.02
        model_line_params = 'GTR{rates}+I{pinv}+G{alpha}+F{freq}'.format(rates="{{{0}}}".format("/".join(rates)),
                                                                         pinv="{{{0}}}".format(pinv),
                                                                         alpha="{{{0}}}".format(alpha),
                                                                         freq="{{{0}}}".format("/".join(freq)))
        # number_of_random_trees = '{' + str(total_trees // 2) + '}'
        # number_of_pars_trees = '{' + str(total_trees - (total_trees // 2)) + '}'
        total_trees = '{' + str(total_trees) + '}'
        p = call(
            [SC.RAXML_NG_SCRIPT, '--msa', path_to_msa, '--threads', '1',
             '--model', model_line_params, '--tree', f'rand{total_trees}', '--redo'], # pars{number_of_pars_trees},rand{number_of_random_trees}
            stdout=PIPE, stdin=PIPE, stderr=STDOUT)
        file_path_base = SC.PATH_TO_RAW_TREE_DATA / data_set
        files_to_delete = ['masked_species_real_msa.phy.raxml.rba', 'masked_species_real_msa.phy.raxml.startTree',
                           'masked_species_real_msa.phy.raxml.reduced.phy', 'masked_species_real_msa.phy.raxml.mlTrees',
                           'masked_species_real_msa.phy.raxml.bestModel', 'masked_species_real_msa.phy.raxml.log',
                           'masked_species_real_msa.phy.raxml.bestTreeCollapsed']
        for file in files_to_delete:
            path_to_delete = str(file_path_base / file)
            if os.path.exists(path_to_delete):
                os.remove(path_to_delete)

    def generate_raxml_max_liklihood_tree_from_specific_start(self, data_set, start_tree_path):
        path_to_msa = str(SC.PATH_TO_RAW_TREE_DATA / data_set / SC.MSA_FILE_NAME)
        stat_path = str(SC.PATH_TO_RAW_TREE_DATA / data_set / SC.PHYML_PARAM_FILE_NAME)
        freq, rates, pinv, alpha = FGT.get_likelihood_params(stat_path)
        alpha = alpha if float(alpha) > 0.02 else 0.02
        model_line_params = 'GTR{rates}+I{pinv}+G{alpha}+F{freq}'.format(rates="{{{0}}}".format("/".join(rates)),
                                                                         pinv="{{{0}}}".format(pinv),
                                                                         alpha="{{{0}}}".format(alpha),
                                                                         freq="{{{0}}}".format("/".join(freq)))
        p = Popen(
            [SC.RAXML_NG_SCRIPT, '--msa', path_to_msa, '--threads', '1',
             '--model', model_line_params, '--tree', str(start_tree_path), '--redo',
             ],
            stdout=PIPE, stdin=PIPE, stderr=STDOUT)
        file_path_base = SC.PATH_TO_RAW_TREE_DATA / data_set
        files_to_delete = ['masked_species_real_msa.phy.raxml.rba', 'masked_species_real_msa.phy.raxml.startTree',
                           'masked_species_real_msa.phy.raxml.reduced.phy', 'masked_species_real_msa.phy.raxml.mlTrees',
                           'masked_species_real_msa.phy.raxml.bestModel', 'masked_species_real_msa.phy.raxml.log',
                           'masked_species_real_msa.phy.raxml.bestTreeCollapsed']
        for file in files_to_delete:
            path_to_delete = str(file_path_base / file)
            if os.path.exists(path_to_delete):
                os.remove(path_to_delete)

        resulting_tree_path = SC.PATH_TO_RAW_TREE_DATA / data_set / SC.RAXML_ML_TREE_FILE_NAME
        counter = 0
        while not resulting_tree_path.is_file():
            time.sleep(2)
            counter += 1
            if counter > 20:
                raise Exception(f'RAXML failed to create best tree - {resulting_tree_path} ')
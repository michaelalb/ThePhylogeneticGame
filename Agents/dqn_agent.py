import gc
import os
import random
import sys
import traceback

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

import SharedConsts as SC
from Agents import dqn_agent_plotting_helper
from Agents.greedy_agent import CheatingGreedyAgent
from Qnetwork import QNetwork
from Qnetwork.ReplayBuffer import ReplayBuffer
from SPR_generator.SPR_move import generate_tree_object
from SharedConsts import EXPLORATION_POLICY, TIMES_TO_LEARN, BATCH_SIZE, TEST_EVERY, BUFFER_SIZE, \
    SAVE_WEIGHTS_EVERY, DTYPE, HORIZON, SHOULD_OPTIMIZE_BRANCH_LENGTHS


class DqnAgent:
    """
    an agent to learn the optimal policy (NOT the hill climb heuristic)
    using a trained neural-net, q-learning and our state_action-action features
    """

    def __init__(self, env, test_env, experiment_unique_dir_name, policy=EXPLORATION_POLICY):
        """
        accept env object and initialize agent and policy.
        we might want to use different envs to train and test the agent,
        so env is created orthogonally to agent
        """
        self.env = env
        self.test_env = test_env
        self.experiment_unique_dir_name = experiment_unique_dir_name
        self.Q_network = QNetwork.QNetwork(env.get_state_action_size(), self.experiment_unique_dir_name)
        self.policy = policy
        self.persistent_testing_results = self.init_testing_logs()

    @staticmethod
    def get_max_action(states_action_pairs_Q_predictions):
        """
        Get the greedy action with respect to the Q value predictions
        :param states_action_pairs_Q_predictions: vector (tensor) of values (predictions)
        :return: chosen action q-val, index of chosen action
        """
        # choose action according to policy
        best_Q_estimation, best_index = torch.max(states_action_pairs_Q_predictions, 0)
        return best_Q_estimation, int(best_index)

    def make_epsilon_greedy_move(self, old_state_actions):
        # choose whether to do a greedy move or a random one
        is_random_step = random.random() < self.policy.next_value()

        if is_random_step:
            # get random move features and Q prediction
            chosen_action_index = random.randint(0, len(old_state_actions) - 1)
            Q_prediction = self.Q_network.predict(old_state_actions[
                                                      chosen_action_index])
        else:
            states_action_pairs_Q_predictions = self.Q_network.predict(old_state_actions)
            Q_prediction, chosen_action_index = self.get_max_action(states_action_pairs_Q_predictions)

        return Q_prediction, chosen_action_index

    def make_softmax_move(self, old_state_actions):
        Q_predictions = self.Q_network.predict(old_state_actions)
        temperature = self.policy.next_value()
        probabilities = torch.softmax(Q_predictions / temperature, dim=0).squeeze()
        assert abs(1 - torch.sum(probabilities)) < 0.001  # assert almost equal 1
        assert len(probabilities.shape) == 1  # 1 dim tensor

        chosen_action_index = probabilities.multinomial(num_samples=1, replacement=True)
        Q_prediction = self.Q_network.predict(old_state_actions[chosen_action_index])

        return Q_prediction, int(chosen_action_index)

    def make_policy_move(self, old_state_actions):

        if self.policy.is_epsilon_greedy:
            Q_prediction, chosen_action_index = self.make_epsilon_greedy_move(old_state_actions)
        else:
            Q_prediction, chosen_action_index = self.make_softmax_move(old_state_actions)

        return Q_prediction, chosen_action_index

    def episode(self):
        """
        orchestrator function for an episode
        """
        # reset env for new episode
        self.env.reset()

        done = False
        first_step = True
        rewards, start_q_val, score_reward = [], 0, 0
        old_state_actions = None

        while not done:

            if old_state_actions is None:
                assert first_step
                old_state_actions = self.env.get_all_neighbor_states()

            Q_prediction, chosen_action_index = self.make_policy_move(old_state_actions)
            state_action = old_state_actions[chosen_action_index]
            # act and get feedback from env (also maybe train your Q network if you like)
            reward, done = self.env.step(chosen_action_index)

            # find maximal next state-action
            new_state_actions = self.env.get_all_neighbor_states()
            new_states_action_predictions = self.Q_network.predict(new_state_actions)
            new_Q_prediction, new_chosen_index = self.get_max_action(new_states_action_predictions)
            new_state_action = new_state_actions[new_chosen_index]

            self.Q_network.add_memory(state_action, reward, done, new_state_action)
            old_state_actions = new_state_actions

            # log stuff
            if first_step:
                start_q_val = Q_prediction.item()
                first_step = False
            rewards.append(reward)

        return sum(rewards), start_q_val

    def train(self, result_dir, episodes=5, save_weights_every=SAVE_WEIGHTS_EVERY):
        """
        larger loop to orchestrate training.
        :return:
        """
        all_episode_reward, all_episode_q_values, loss = [], [], []
        try:
            for episode in range(episodes):
                gc.collect()
                cumulative_reward, start_q_val = self.episode()
                all_episode_reward.append(cumulative_reward)
                all_episode_q_values.append(start_q_val)

                if episode % save_weights_every == 0:
                    self.Q_network.save_all(episode)

                # for every added memory - learn TIMES_TO_LEARN times
                if len(self.Q_network.memory) > BATCH_SIZE:
                    for _ in range(TIMES_TO_LEARN):
                        loss.append(self.Q_network.learn())

                # plot
                if episode % 250 == 0 or episode < 20:
                    self.env.helper.plot_rl_scores(result_dir, all_episode_reward, all_episode_q_values, loss)

                # test agent
                if episode % TEST_EVERY == 0 and episode > 1:
                    self.test(result_dir=result_dir, episode=episode)

        # this is for catching and reporting about assertion errors that might occur during runtime
        except AssertionError:
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb)  # Fixed format
            tb_info = traceback.extract_tb(tb)
            filename, line, func, text = tb_info[-1]

            print('An error occurred on line {} in statement {}'.format(line, text))
        finally:
            self.Q_network.prepare_to_exit(episodes)
            self.env.remove_files()

        return all_episode_reward, all_episode_q_values, loss

    def test(self, result_dir, episode, num_of_tests=20, final_test=False):

        # track on which episodes were testing on

        # create dirs
        train_set_dir, test_set_dir, test_summary_dir, train_summary_dir = DqnAgent.create_episode_testing_dirs(
            result_dir=result_dir, episode=episode)

        # do the actual testing
        raw_results_test_set_df, raw_results_train_set_df, train_full_travel_logs, test_full_travel_logs,\
        train_amazing_moves, test_amazing_moves,  train_hill_climb_travel_logs, test_hill_climb_travel_logs\
            = self.run_test_episodes(num_of_tests=num_of_tests)

        # adding results that need persisting across episodes to list:
        self.add_results_to_episode_testing_data_log(raw_results_train_set_df, raw_results_test_set_df, episode)

        # create testing aggregate plots
        self.plot_and_log_results(raw_results_test_set_df=raw_results_test_set_df,
                                  raw_results_train_set_df=raw_results_train_set_df, result_dir=result_dir,
                                  test_summary_dir=test_summary_dir,
                                  train_summary_dir=train_summary_dir,
                                  train_set_dir=train_set_dir, test_set_dir=test_set_dir,
                                  train_full_travel_logs=train_full_travel_logs,
                                  test_full_travel_logs=test_full_travel_logs,
                                  train_amazing_moves=train_amazing_moves,
                                  test_amazing_moves=test_amazing_moves,
                                  train_hill_climb_travel_logs=train_hill_climb_travel_logs,
                                  test_hill_climb_travel_logs=test_hill_climb_travel_logs
                                  )

        # save raw results
        raw_results_test_set_df.to_csv(test_set_dir / 'raw_results_test_set.csv', index=False)
        raw_results_train_set_df.to_csv(train_set_dir / 'raw_results_train_set.csv', index=False)

        if final_test:
            self.test_env.remove_files()
            self.env.remove_files()

        return self.persistent_testing_results['test_diff_bests'][-1], \
               self.persistent_testing_results['test_improvement_percent'][-1]

    def generate_test_results(self, test_env, number_of_tests):
        """
        tests agent on all datasets in test_env, test_env is just a var name- it's ok to send the train env
        """
        # save the old env
        old_env = self.env
        self.Q_network.to_eval_mode()

        self.env = test_env
        results_per_data_set = {}

        for dataset in self.env.helper.all_data_sets:
            # set up constants per data set:
            # get raxml benchmark per dataset - this is a static number currently, if we decide to re-calc every test
            # move into inner loop and change const to list
            self.env.reset(dataset=dataset)
            # normalization factor for dll (NJ tree -likelihood)
            specific_norm_factor_ll = abs(self.env.helper.get_normalization_factor_per_ds(data_set=dataset))
            results_per_data_set[dataset] = {'ll_results': [], 'rf_results': [],
                                             'dynamic_raxml_best_ll': [],
                                             'specific_norm_factor_ll': specific_norm_factor_ll,
                                             'full_ll_travel_logs': [], 'hill_climb_travel_logs': [],
                                             'amazing_moves': [], 'init_likelihood': []}

            for test_replication in range(number_of_tests):
                # create condition for full logging - time-consuming
                verbose = test_replication == 0
                # reset env starting tree
                starting_tree = self.env.reset(dataset=dataset)
                current_tree_raxml_best_ll = self.env.get_raxml_likelihood_specific_starting_tree(starting_tree)
                init_likelihood = self.env.likelihood
                # run actual test episode
                gc.collect()
                max_likelihood_reached, agent_ll_log, agent_rf_log, amazing_moves = self.run_test_episode(
                    dataset, current_tree_raxml_best_ll, should_return_full_travel_log=verbose)

                # log results
                # if you want to add rf - do it here
                results_per_data_set[dataset]['dynamic_raxml_best_ll'].append(current_tree_raxml_best_ll)
                results_per_data_set[dataset]['ll_results'].append(max_likelihood_reached)
                results_per_data_set[dataset]['init_likelihood'].append(init_likelihood)
                if verbose:
                    results_per_data_set[dataset]['full_ll_travel_logs'].append(agent_ll_log)
                    # log hill-climb agent
                    hill_climb_log = CheatingGreedyAgent.hill_climb(dataset, starting_tree)
                    results_per_data_set[dataset]['hill_climb_travel_logs'].append(hill_climb_log)
                if len(amazing_moves) > 0:
                    results_per_data_set[dataset]['amazing_moves'] += amazing_moves

        # load the old env
        self.env = old_env
        assert self.env.is_train
        self.Q_network.to_train_mode()

        return results_per_data_set

    def run_test_episode(self, dataset, raxml_best_ll, should_return_full_travel_log=False,
                         should_calculate_rf=False):
        # set up variables
        # if should_return_full_travel_log:
        if should_return_full_travel_log:
            agent_ll_log = [self.env.likelihood]
            agent_rf_log = [self.env.tree] if should_calculate_rf else []
        else:
            agent_ll_log, agent_rf_log = [[], []]
        done = False
        amazing_moves = []
        move_counter = 0

        while not done:
            move_counter += 1
            state_actions = self.env.get_all_neighbor_states()
            states_action_pairs_Q_predictions = self.Q_network.predict(state_actions)
            Q_prediction, chosen_action_index = self.get_max_action(states_action_pairs_Q_predictions)

            # act and get feedback from env
            should_perform_actual_reward_calc = SHOULD_OPTIMIZE_BRANCH_LENGTHS or move_counter >= HORIZON - 1 # should_return_full_travel_log
            if should_perform_actual_reward_calc or should_return_full_travel_log:
                reward, done = self.env.step(chosen_action_index)
                likelihood_from_env = self.env.likelihood
                agent_ll_log.append(likelihood_from_env)
                # used to calc rf dist
                if should_calculate_rf:
                    agent_rf_log.append(self.env.tree)
            else:
                reward, done = self.env.step(chosen_action_index, should_calculate_reward=False)
                likelihood_from_env = float('-inf')

            if should_perform_actual_reward_calc and likelihood_from_env >= raxml_best_ll:
                current_amazing_move_dict = {
                    'move_index': move_counter,
                    'horizon': HORIZON,
                    'raxml_best_ll': raxml_best_ll,
                    'agents_ll': likelihood_from_env,
                    'data_set': dataset,
                    'is_better': likelihood_from_env > raxml_best_ll
                }
                amazing_moves.append(current_amazing_move_dict)

        # our horizon could force the agent to preform a last bad move
        assert len(agent_ll_log) > 1
        max_likelihood_reached = max(agent_ll_log[-1], agent_ll_log[-2])

        return max_likelihood_reached, agent_ll_log, agent_rf_log, amazing_moves

    @staticmethod
    def rf_metric(dataset, agent_travel_log, rf_norms_per_ds, verbose, test_result_dir):
        """Edits rf_norms_per_ds inplace, no need to return result"""

        # RF metric - to reduce test time. remove and uncomment above if you wish to get RF metric
        ml_tree_path = SC.PATH_TO_RAW_TREE_DATA / dataset / SC.RAXML_ML_TREE_FILE_NAME
        ml_tree_obj = generate_tree_object(ml_tree_path)
        t_last, t_before_last = agent_travel_log[1][-1], agent_travel_log[1][-2]
        # norm_rf (normalized by the max_rf). This is equ to 2n-6.
        specific_norm_factor_rf = ml_tree_obj.robinson_foulds(t_last, unrooted_trees=True)[1]
        # to know the max rf possible take 1
        rf_distance = min(ml_tree_obj.robinson_foulds(t_last, unrooted_trees=True)[0],
                          ml_tree_obj.robinson_foulds(t_before_last, unrooted_trees=True)[0])
        rf_distance /= specific_norm_factor_rf
        rf_norms_per_ds[dataset] = (rf_distance, specific_norm_factor_rf)

        # plot line plot of the first replicate run (only)
        if verbose:
            dqn_agent_plotting_helper.plot_specific_agent_run_all_moves_rf(agent_reached_trees=agent_travel_log[1],
                                                                           ml_tree_obj=ml_tree_obj,
                                                                           date_set=dataset,
                                                                           target_dir=test_result_dir)

    def learn_from_experience(self, unique_dir_name_lst, result_dir, processed_data_dir_name, epochs):
        """
        trains the agent on given experience without interacting with env or gaining any new exp
        :param processed_data_dir_name: dir to save the processed data in - if data exists
        in this folder - this becomes the training data
        :param epochs:
        :param result_dir:
        :param unique_dir_name_lst: list of dirs where memory buffer files from previous agent runs can be found
        :return: None
        """
        save_weights_every = epochs // 20 + 1
        dataloader = self.get_data(unique_dir_name_lst, processed_data_dir_name)

        epoch_loss = []
        for epoch in range(epochs):

            batch_loss = []
            for state_actions, rewards, dones, next_state_actions in dataloader:  # returns batches
                loss = self.Q_network.learn_on_batch(state_actions, rewards, dones, next_state_actions)
                batch_loss.append(loss)

            epoch_loss.append(np.mean(batch_loss))  # log average loss of epoch

            # plot every epoch
            dqn_agent_plotting_helper.plot_loss(result_dir, epoch_loss)

            if epoch % save_weights_every == 0:
                self.Q_network.save_weights(episode=epoch)
            # test agent
            if epoch % TEST_EVERY == 0 and epoch > 1:
                self.test(result_dir=result_dir, episode=epoch)

        self.Q_network.save_weights(episode=epochs)

    @staticmethod
    def get_buffers(unique_dir_name_lst):
        # get a list of all Replay memories
        replay_buffers = []
        for dir_name in unique_dir_name_lst:
            memory_path = SC.EXPERIMENTS_RESDIR / dir_name / SC.REPLAY_BUFFER_FILE_NAME
            memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, memory_path=memory_path)
            replay_buffers.append(memory)

        return replay_buffers

    def combine_buffers(self, replay_buffers):

        data = []
        for buffer in replay_buffers:
            data.extend(list(buffer.memory))

        state_actions, rewards, dones, next_state_actions = self.Q_network.memory.process_experience(data)
        data_tuple = state_actions, rewards, dones, next_state_actions

        return data_tuple

    def get_data(self, unique_dir_name_lst, processed_data_dir_name):
        sa, rewards, dones, next_sa = self.read_data(unique_dir_name_lst, processed_data_dir_name)
        dataset = TensorDataset(sa, rewards, dones, next_sa)

        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    def read_data(self, unique_dir_name_lst, processed_data_dir_name):
        data_dir = SC.EXPERIMENTS_RESDIR / processed_data_dir_name
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        data_path = data_dir / SC.PROCESSED_DATA_FILE_NAME

        try:
            # if file exists - take it
            data = torch.load(data_path)
            state_actions, rewards, dones, next_state_actions = self.to_gpu(data)
            print('Found processed data - ignoring buffer_data_dir_lst')
        except FileNotFoundError:
            print('No processed data - reading buffer_data_dir_lst')
            replay_buffers = self.get_buffers(unique_dir_name_lst)
            data = self.combine_buffers(replay_buffers)
            torch.save(data, data_path)
            state_actions, rewards, dones, next_state_actions = data

        return state_actions, rewards, dones, next_state_actions

    @staticmethod
    def to_gpu(data):
        device = torch.device("cuda" if SC.USE_CUDA else "cpu")
        dtype = DTYPE
        state_actions, rewards, dones, next_state_actions = data

        state_actions = state_actions.to(device=device, dtype=dtype)
        rewards = rewards.to(device=device, dtype=dtype)
        next_state_actions = next_state_actions.to(device=device, dtype=dtype)
        dones = dones.to(device=device, dtype=dtype)

        return state_actions, rewards, dones, next_state_actions

    def get_random_data(self, unique_dir_name_lst, processed_data_dir_name):
        sa, rewards, dones, next_sa = self.read_data(unique_dir_name_lst, processed_data_dir_name)

        sa = torch.rand_like(sa)
        next_sa = torch.rand_like(next_sa)
        dataset = TensorDataset(sa, rewards, dones, next_sa)

        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    @staticmethod
    def create_episode_testing_dirs(result_dir, episode):
        test_result_dir = result_dir / SC.RESULTS_FOLDER_NAME / 'test_results_{}'.format(episode)
        test_set_dir = test_result_dir / 'test_set'
        train_set_dir = test_result_dir / 'train_set'

        test_summary_dir = result_dir / 'results_and_graphs' / 'test_summaries'
        train_summary_dir = result_dir / 'results_and_graphs' / 'train_summaries'

        test_result_dir.mkdir(parents=True, exist_ok=True)
        test_set_dir.mkdir(exist_ok=True)
        train_set_dir.mkdir(exist_ok=True)
        test_summary_dir.mkdir(exist_ok=True)
        train_summary_dir.mkdir(exist_ok=True)

        return train_set_dir, test_set_dir, test_summary_dir, train_summary_dir

    def create_specific_data_set_results_df(self, num_of_tests, data_set, data_set_results):
        tmp_df = pd.DataFrame(columns=["test_replication_number"], data=[[i] for i in range(num_of_tests)])
        tmp_df["data_set_number"] = data_set
        tmp_df["specific_norm_factor_ll"] = data_set_results['specific_norm_factor_ll']
        tmp_df["raxml_ml"] = pd.Series(data_set_results['dynamic_raxml_best_ll'])
        tmp_df["max_ll_reached"] = pd.Series(data_set_results["ll_results"])
        tmp_df['ll_diff_real_scale'] = tmp_df["max_ll_reached"] - tmp_df['raxml_ml']
        tmp_df['ll_diff'] = tmp_df['ll_diff_real_scale'] / tmp_df['specific_norm_factor_ll']
        tmp_df["rf_dist"] = 0
        tmp_df["rf_norm_factor"] = 0

        # calc improvement %
        nj_ll = self.env.get_NJ_likelihood(data_set=data_set)
        init_likelihood = pd.Series(data_set_results["init_likelihood"])
        raxml_improvement = tmp_df["raxml_ml"] - init_likelihood
        our_improvement = tmp_df["max_ll_reached"] - init_likelihood
        nj_improvement = nj_ll - init_likelihood
        tmp_df['improvement_percent'] = our_improvement / raxml_improvement
        tmp_df['improvement_percent_nj'] = nj_improvement / raxml_improvement
        tmp_df['improvement_percent_us_nj'] = our_improvement / nj_improvement
        tmp_df['improvement_percent_raxml_nj'] = raxml_improvement / nj_improvement
        return tmp_df

    def format_test_episodes_results(self, results_per_data_set, num_of_tests):
        raw_results_df = pd.DataFrame(columns=SC.TESTING_RESULT_COLUMNS)
        amazing_moves_log, full_travel_logs, hill_climb_travel_logs = {}, {}, {}
        for data_set, data_set_results in results_per_data_set.items():
            tmp_df = self.create_specific_data_set_results_df(num_of_tests, data_set, data_set_results)
            raw_results_df = raw_results_df.append(tmp_df, ignore_index=True)
            # return full travel logs and amazing moves
            if len(data_set_results.get('full_ll_travel_logs', [])) > 0:
                full_travel_logs[data_set] = data_set_results.get('full_ll_travel_logs')
                hill_climb_travel_logs[data_set] = data_set_results.get('hill_climb_travel_logs')
            if len(data_set_results.get('amazing_moves', [])) > 0:
                amazing_moves_log[data_set] = data_set_results.get('amazing_moves')

        return raw_results_df, full_travel_logs, amazing_moves_log, hill_climb_travel_logs

    def run_test_episodes(self, num_of_tests):
        # run tests for train environment
        results_per_data_set_train = self.generate_test_results(test_env=self.env, number_of_tests=num_of_tests)
        # run tests for test environment
        results_per_data_set_test = self.generate_test_results(test_env=self.test_env, number_of_tests=num_of_tests)

        # sort and append results train
        raw_results_train_set_df, train_full_travel_logs, train_amazing_moves, train_hill_climb_travel_logs\
            = self.format_test_episodes_results(results_per_data_set_train, num_of_tests)
        raw_results_test_set_df, test_full_travel_logs, test_amazing_moves, test_hill_climb_travel_logs\
            = self.format_test_episodes_results(results_per_data_set_test, num_of_tests)

        return raw_results_test_set_df, raw_results_train_set_df, train_full_travel_logs, test_full_travel_logs,\
               train_amazing_moves, test_amazing_moves, train_hill_climb_travel_logs, test_hill_climb_travel_logs

    @staticmethod
    def init_testing_logs():
        logs = {'episodes': [], 'train_diff_means_norm': [], 'train_diff_bests_norm': [], 'test_diff_means_norm': [],
                'test_diff_bests_norm': [], 'train_diff_means': [], 'train_diff_bests': [], 'test_diff_means': [],
                'test_diff_bests': [], 'per_data_set': {},
                'train_improvement_percent': [], 'test_improvement_percent': [],
                'train_improvement_percent_us_nj': [], 'train_improvement_percent_nj_raxml': [],
                'train_improvement_percent_raxml_nj': [], 'test_improvement_percent_us_nj': [],
                'test_improvement_percent_nj_raxml': [], 'test_improvement_percent_raxml_nj': []}
        return logs

    def add_results_to_episode_testing_data_log(self, raw_results_train_set_df, raw_results_test_set_df, episode):
        self.persistent_testing_results['episodes'].append(episode)

        # record results that need saving across episodes
        self.persistent_testing_results['train_diff_means_norm'].append(
            np.mean(raw_results_train_set_df.groupby('data_set_number')['ll_diff'].mean()))
        self.persistent_testing_results['train_diff_bests_norm'].append(
            np.mean(raw_results_train_set_df.groupby('data_set_number')['ll_diff'].max()))
        self.persistent_testing_results['test_diff_means_norm'].append(
            np.mean(raw_results_test_set_df.groupby('data_set_number')['ll_diff'].mean()))
        self.persistent_testing_results['test_diff_bests_norm'].append(
            np.mean(raw_results_test_set_df.groupby('data_set_number')['ll_diff'].max()))

        self.persistent_testing_results['train_diff_means'].append(
            np.mean(raw_results_train_set_df.groupby('data_set_number')['ll_diff_real_scale'].mean()))
        self.persistent_testing_results['train_diff_bests'].append(
            np.mean(raw_results_train_set_df.groupby('data_set_number')['ll_diff_real_scale'].max()))
        self.persistent_testing_results['test_diff_means'].append(
            np.mean(raw_results_test_set_df.groupby('data_set_number')['ll_diff_real_scale'].mean()))
        self.persistent_testing_results['test_diff_bests'].append(
            np.mean(raw_results_test_set_df.groupby('data_set_number')['ll_diff_real_scale'].max()))

        # record % improvement us compared to raxml
        self.persistent_testing_results['train_improvement_percent'].append(
            np.mean(raw_results_train_set_df.groupby('data_set_number')['improvement_percent'].max()))
        self.persistent_testing_results['test_improvement_percent'].append(
            np.mean(raw_results_test_set_df.groupby('data_set_number')['improvement_percent'].max()))

        # record % improvement us compared to nj
        self.persistent_testing_results['train_improvement_percent_us_nj'].append(
            np.mean(raw_results_train_set_df.groupby('data_set_number')['improvement_percent_us_nj'].max()))
        self.persistent_testing_results['test_improvement_percent_us_nj'].append(
            np.mean(raw_results_test_set_df.groupby('data_set_number')['improvement_percent_us_nj'].max()))

        # record % improvement nj compared to raxml
        self.persistent_testing_results['train_improvement_percent_nj_raxml'].append(
            np.mean(raw_results_train_set_df.groupby('data_set_number')['improvement_percent_nj'].max()))
        self.persistent_testing_results['test_improvement_percent_nj_raxml'].append(
            np.mean(raw_results_test_set_df.groupby('data_set_number')['improvement_percent_nj'].max()))

        # record % improvement raxml compared to nj
        self.persistent_testing_results['train_improvement_percent_raxml_nj'].append(
            np.mean(raw_results_train_set_df.groupby('data_set_number')['improvement_percent_raxml_nj'].max()))
        self.persistent_testing_results['test_improvement_percent_raxml_nj'].append(
            np.mean(raw_results_test_set_df.groupby('data_set_number')['improvement_percent_raxml_nj'].max()))

        comb_res = pd.concat([raw_results_test_set_df, raw_results_train_set_df])
        for data_set in comb_res['data_set_number'].unique():
            if data_set not in self.persistent_testing_results['per_data_set']:
                self.persistent_testing_results['per_data_set'][data_set] = {}
                self.persistent_testing_results['per_data_set'][data_set]['normalized'] = {'bests': [],
                                                                                           'means': [], 'all': []}
                self.persistent_testing_results['per_data_set'][data_set]['at_scale'] = {'bests': [],
                                                                                         'means': [], 'all': []}
            relevant_res = comb_res[comb_res['data_set_number'] == data_set]
            self.persistent_testing_results['per_data_set'][data_set]['normalized']['bests'].append(
                np.max(relevant_res['ll_diff']))
            self.persistent_testing_results['per_data_set'][data_set]['normalized']['means'].append(
                np.mean(relevant_res['ll_diff']))
            self.persistent_testing_results['per_data_set'][data_set]['normalized']['all'] += list(
                relevant_res['ll_diff'].values)
            self.persistent_testing_results['per_data_set'][data_set]['at_scale']['bests'].append(
                np.max(relevant_res['ll_diff_real_scale']))
            self.persistent_testing_results['per_data_set'][data_set]['at_scale']['means'].append(
                np.mean(relevant_res['ll_diff_real_scale']))
            self.persistent_testing_results['per_data_set'][data_set]['at_scale']['all'] += list(
                relevant_res['ll_diff_real_scale'].values)
            self.persistent_testing_results['per_data_set'][data_set]['raxml'] = list(relevant_res['raxml_ml'])

    def plot_and_log_results(self, raw_results_test_set_df, raw_results_train_set_df, result_dir,
                             test_summary_dir, train_summary_dir, train_set_dir, test_set_dir,
                             train_full_travel_logs, test_full_travel_logs, train_amazing_moves, test_amazing_moves,
                             train_hill_climb_travel_logs, test_hill_climb_travel_logs):

        episode_list = self.persistent_testing_results.get('episodes')

        # plot main raxml- diff plot - normalized and not normalized
        dqn_agent_plotting_helper.create_raxml_diff_agg_plot(target_dir=result_dir,
                                                             episode_list=episode_list,
                                                             train_diff_mean=self.persistent_testing_results.get(
                                                                 'train_diff_means_norm'),
                                                             test_diff_mean=self.persistent_testing_results.get(
                                                                 'test_diff_means_norm'),
                                                             train_diff_best=self.persistent_testing_results.get(
                                                                 'train_diff_bests_norm'),
                                                             test_diff_best=self.persistent_testing_results.get(
                                                                 'test_diff_bests_norm'))

        dqn_agent_plotting_helper.create_raxml_diff_agg_plot(target_dir=result_dir,
                                                             episode_list=episode_list,
                                                             train_diff_mean=self.persistent_testing_results.get(
                                                                 'train_diff_means'),
                                                             test_diff_mean=self.persistent_testing_results.get(
                                                                 'test_diff_means'),
                                                             train_diff_best=self.persistent_testing_results.get(
                                                                 'train_diff_bests'),
                                                             test_diff_best=self.persistent_testing_results.get(
                                                                 'test_diff_bests'),
                                                             is_normalized=False)

        dqn_agent_plotting_helper.create_improvement_percent_from_starting(
            target_dir=result_dir, episode_list=episode_list,
            nj_improvement_from_raxml_train=self.persistent_testing_results.get('train_improvement_percent_nj_raxml'),
            nj_improvement_from_raxml_test=self.persistent_testing_results.get('test_improvement_percent_nj_raxml'),
            our_improvement_from_raxml_train=self.persistent_testing_results.get('train_improvement_percent'),
            our_improvement_from_raxml_test=self.persistent_testing_results.get('test_improvement_percent'))

        dqn_agent_plotting_helper.create_improvement_percent_from_nj(
            target_dir=result_dir, episode_list=episode_list,
            raxml_improvement_from_nj_train=self.persistent_testing_results.get('train_improvement_percent_raxml_nj'),
            raxml_improvement_from_nj_test=self.persistent_testing_results.get('test_improvement_percent_raxml_nj'),
            our_improvement_from_nj_train=self.persistent_testing_results.get('train_improvement_percent_us_nj'),
            our_improvement_from_nj_test=self.persistent_testing_results.get('test_improvement_percent_us_nj'))

        # plot aggregations per data set
        self.plot_aggregations_per_dataset(raw_results_set_df=raw_results_test_set_df, summary_dir=test_summary_dir,
                                           episode_list=episode_list,
                                           target_dir=test_set_dir, full_travel_logs=test_full_travel_logs,
                                           hill_climb_travel_logs=test_hill_climb_travel_logs)
        self.plot_aggregations_per_dataset(raw_results_set_df=raw_results_train_set_df, summary_dir=train_summary_dir,
                                           episode_list=episode_list,
                                           target_dir=train_set_dir, full_travel_logs=train_full_travel_logs,
                                           hill_climb_travel_logs=train_hill_climb_travel_logs)

        dqn_agent_plotting_helper.log_amazing_moves(amazing_moves={**train_amazing_moves, **test_amazing_moves},
                                                    target_dir=result_dir)

        # plot all points test\train - normalized
        dqn_agent_plotting_helper.create_ramxl_diff_all_points_plot(raw_results_df=raw_results_train_set_df,
                                                                    target_dir=train_set_dir)
        dqn_agent_plotting_helper.create_ramxl_diff_all_points_plot(raw_results_df=raw_results_test_set_df,
                                                                    target_dir=test_set_dir)

        # plot all points test\train - at scale
        dqn_agent_plotting_helper.create_ramxl_diff_all_points_plot(raw_results_df=raw_results_train_set_df,
                                                                    target_dir=train_set_dir, is_normalized=False)
        dqn_agent_plotting_helper.create_ramxl_diff_all_points_plot(raw_results_df=raw_results_test_set_df,
                                                                    target_dir=test_set_dir, is_normalized=False)

    def plot_aggregations_per_dataset(self, raw_results_set_df, summary_dir, episode_list,
                                      target_dir, full_travel_logs, hill_climb_travel_logs):
        for data_set in raw_results_set_df['data_set_number'].unique():
            dqn_agent_plotting_helper.create_raxml_diff_per_data_set_plot(target_dir=summary_dir,
                                                                          episode_list=episode_list,
                                                                          bests=self.persistent_testing_results[
                                                                              'per_data_set'][data_set]['normalized'][
                                                                              'bests'],
                                                                          means=self.persistent_testing_results[
                                                                              'per_data_set'][data_set]['normalized'][
                                                                              'means'],
                                                                          data_set=data_set)
            dqn_agent_plotting_helper.create_raxml_diff_per_data_set_plot(target_dir=summary_dir,
                                                                          episode_list=episode_list,
                                                                          bests=self.persistent_testing_results[
                                                                              'per_data_set'][data_set]['at_scale'][
                                                                              'bests'],
                                                                          means=self.persistent_testing_results[
                                                                              'per_data_set'][data_set]['at_scale'][
                                                                              'means'],
                                                                          data_set=data_set,
                                                                          is_normalized=False)
            for i, travel_log in enumerate(full_travel_logs.get(data_set)):
                hill_climb_results = hill_climb_travel_logs.get(data_set)[i]
                raxml_best_ll = self.persistent_testing_results['per_data_set'][data_set]['raxml'][i]
                dqn_agent_plotting_helper.plot_specific_agent_run_all_moves(results=travel_log,
                                                                            hill_climb_results=hill_climb_results,
                                                                            travel_log_index=i,
                                                                            raxml_best_ll=raxml_best_ll,
                                                                            target_dir=target_dir,
                                                                            date_set=data_set)

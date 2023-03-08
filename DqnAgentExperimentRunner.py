import os
import random

import pandas as pd

import SharedConsts as sc
from Agents.dqn_agent import DqnAgent
from Reinforcement_env.env import PhyloGame


def run_dqn_agent_experiment(**kwargs):
    agent, result_dir = set_up_dqn_agent(kwargs['cpus'], kwargs['experiment_unique_dir_name'])

    all_episode_rewards, all_episode_q, loss = agent.train(episodes=sc.EPISODES, result_dir=result_dir)

    # change to show=False when running jobs (then it saves the plot in the dataset dir)
    agent.env.helper.plot_rl_scores(result_dir, all_episode_rewards, all_episode_q, loss)

    agent.test(result_dir=result_dir, final_test=True, episode=sc.EPISODES)


def set_up_dqn_agent(cpus, experiment_unique_dir_name, buffer_data_dir_lst=None):
    # buffer_data_dir_lst - should only be set if this is a learn from experience scenario
    random.seed(sc.RANDOM_SEED)
    number_of_cpus = int(cpus) if cpus is not None else None

    result_dir = sc.EXPERIMENTS_RESDIR / experiment_unique_dir_name
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    train_datasets, test_datasets = choose_rl_datasets(sc.NUMBER_OF_SPECIES, sc.HOW_MANY_DATASETS_TRAIN)
    print(f'train datasets: {train_datasets}')
    print(f'test datasets: {test_datasets}')
    # train_datasets = switch_train_sets(buffer_data_dir_lst, train_datasets)
    env = PhyloGame(datasets=train_datasets, number_of_cpus_to_use=number_of_cpus,
                    use_random_starts=sc.USE_RANDOM_STARTING_TREES, results_dir=result_dir, is_train=True)
    test_env = PhyloGame(datasets=test_datasets, number_of_cpus_to_use=number_of_cpus,
                         use_random_starts=sc.USE_RANDOM_STARTING_TREES, results_dir=result_dir, is_train=False)

    agent = DqnAgent(env=env, test_env=test_env, experiment_unique_dir_name=experiment_unique_dir_name)
    return agent, result_dir


def choose_rl_datasets(number_of_species, how_many_ds_train):
    if sc.FIXED_TEST_DATASETS is None:
        test_df = pd.read_csv(sc.PATH_TO_TESTING_TREES_FILE)
        test_datasets = list(test_df[test_df['ntaxa'].isin(number_of_species)]['data_set_number'].astype(str))
    else:
        test_datasets = sc.FIXED_TEST_DATASETS

    if sc.FIXED_TRAIN_DATASETS is None:
        train_df = pd.read_csv(sc.PATH_TO_TRAINING_TREES_FILE)
        train_datasets = list(train_df[train_df['ntaxa'].isin(number_of_species)]['data_set_number'].astype(str))
        train_datasets = random.sample(train_datasets, how_many_ds_train)
    else:
        train_datasets = sc.FIXED_TRAIN_DATASETS

    return train_datasets, test_datasets

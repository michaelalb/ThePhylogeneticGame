import os
import json
import random
import pandas as pd
import SharedConsts as sc
import matplotlib.pyplot as plt

from Agents.dqn_agent import DqnAgent
from Reinforcement_env.env import PhyloGame
from Agents.Qnetwork import Qnet_parameters as qp
from Agents.random_walk_agent import RandomWalkAgent
from Runners.RunnerToolBox import change_permissions, choose_rl_datasets


def run_dqn_agent_experiment(**kwargs):
    agent, result_dir = set_up_dqn_agent(kwargs['cpus'], kwargs['experiment_unique_dir_name'])

    all_episode_rewards, all_episode_q, loss = agent.train(episodes=qp.EPISODES, result_dir=result_dir)

    # change to show=False when running jobs (then it saves the plot in the dataset dir)
    agent.env.helper.plot_rl_scores(result_dir, all_episode_rewards, all_episode_q, loss)

    agent.test(result_dir=result_dir, final_test=True, episode=qp.EPISODES)
    change_permissions(paths_to_permit=[result_dir, sc.RL_EXPERIMENTS_LOGGER_PATH, sc.CODE_PATH])


def learn_from_experience(buffer_data_dir_lst, **kwargs):
    agent, result_dir = set_up_dqn_agent(kwargs['cpus'], kwargs['experiment_unique_dir_name'], buffer_data_dir_lst)

    agent.learn_from_experience(unique_dir_name_lst=buffer_data_dir_lst, epochs=qp.EPOCHS, result_dir=result_dir,
                                processed_data_dir_name=kwargs['processed_data_dir_name'])

    change_permissions(paths_to_permit=[result_dir, sc.RL_EXPERIMENTS_LOGGER_PATH, sc.CODE_PATH])


def set_up_dqn_agent(cpus, experiment_unique_dir_name, buffer_data_dir_lst=None):
    # buffer_data_dir_lst - should only be set if this is a learn from experience scenario
    random.seed(qp.RANDOM_SEED)
    number_of_cpus = int(cpus) if cpus is not None else None

    result_dir = sc.EXPERIMENTS_RESDIR / experiment_unique_dir_name
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    train_datasets, test_datasets = choose_rl_datasets(qp.NUMBER_OF_SPECIES, qp.HOW_MANY_DATASETS_TRAIN)

    # train_datasets = switch_train_sets(buffer_data_dir_lst, train_datasets)
    env = PhyloGame(datasets=train_datasets, number_of_cpus_to_use=number_of_cpus,
                    use_random_starts=qp.USE_RANDOM_STARTING_TREES, results_dir=result_dir, is_train=True)
    test_env = PhyloGame(datasets=test_datasets, number_of_cpus_to_use=number_of_cpus,
                         use_random_starts=qp.USE_RANDOM_STARTING_TREES, results_dir=result_dir, is_train=False)

    agent = DqnAgent(env=env, test_env=test_env, experiment_unique_dir_name=experiment_unique_dir_name)
    agent.env.helper.map_unique_name_to_params_log(experiment_unique_dir_name=experiment_unique_dir_name,
                                                   train_datasets=train_datasets, buffer_data_dir_lst=buffer_data_dir_lst)
    return agent, result_dir



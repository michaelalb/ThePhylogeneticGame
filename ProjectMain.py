import argparse

import DqnAgentExperimentRunner as DqnAgentRunner

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Running RL project Main')
    parser.add_argument('--experiment_unique_dir_name', '-dirname', default='oz_and_dana_debug')
    parser.add_argument('--cpus', help='number of cpus allocated to the job', default=1)
    parser.add_argument('--experiment_name', '-name', default='NoName', help='valid for regression net runs')
    parser.add_argument('--processed_data_dir_name', '-data_dir_name', default='stage_5_data',
                        help='data used learn_from_experience')
    parser.add_argument('--datasets', '-datasets', default='None',
                        help='data used learn_from_experience')
    args = parser.parse_args()

    DqnAgentRunner.run_dqn_agent_experiment(**vars(args))

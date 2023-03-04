import argparse

import DqnAgentExperimentRunner as DqnAgentRunner

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Running RL project Main')
    parser.add_argument('--experiment_unique_dir_name', '-dirname', default='oz_and_dana_debug')
    parser.add_argument('--cpus', help='number of cpus allocated to the job', default=1)
    args = parser.parse_args()

    DqnAgentRunner.run_dqn_agent_experiment(**vars(args))

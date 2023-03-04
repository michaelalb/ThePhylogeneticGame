"""
this file is meant for constants shared across all modules
note: only one place for the data, gitcode, results, and any input/output (Michael's homedir/RL).
the temp PyCharm synced code could be in all our users for debugging
"""
from NJ_STATIC_DATA import NJ_TBL_FOR_DATASET_DICT
from pathlib import Path

import Qnetwork.ExploreScheduler as ExploreScheduler
import torch

##########################
#EXPERIMENT CONFIGURATION#
##########################

###################################
# constants that define the model #
###################################
HIDDEN_SIZES = [4096] + [4096] + [2048] + [128] + [32]

###################
# hyperparameters #
###################
BATCH_SIZE = 128
LEARNING_RATE = 1e-5
SPR_RADIUS = None     # if not None, should be > 1


####################################
# Current network train parameters #
####################################
# options for dtype are: float(32-bit), double(64-bit), half(16-bit) default is float
# half only works on gpu
DTYPE = torch.float
LOSS_FUNC = torch.nn.MSELoss()
LOSS_FUNC_NAME = "MSE-Loss"
FIXED_TRAIN_DATASETS = ['balibase_RV12_BB12017']
FIXED_TEST_DATASETS = ['balibase_RV12_BB12017']
num_of_sp = 7
SHOULD_OPTIMIZE_BRANCH_LENGTHS = False
NUMBER_OF_SPECIES = [num_of_sp]  # only one num for now
HOW_MANY_DATASETS_TRAIN = 11 if FIXED_TRAIN_DATASETS is None else len(FIXED_TRAIN_DATASETS)
USE_RANDOM_STARTING_TREES = True
TEST_EVERY = 500
SAVE_WEIGHTS_EVERY = 250
#################
# RL parameters #
#################
TIMES_TO_LEARN = 50  # how often to update the network
BUFFER_SIZE = int(1e4)  # replay buffer size
EPISODES = 5000
RANDOM_SEED = 'xx'
RANDOM_STARTING_TREES_MEMORY_BATCH_FOR_DS = 100
# ###### these are the correct strings to use.
# ALSO NO TRANSFORM MEANS JUST DIVIDING BY (-CURRENT LL)

# horizon params
horizon_dict = {    # if not in dict - error
    7: 20,
    12: 20,
    15: 20,
    20: 30,
    70: 50
}
HORIZON = horizon_dict[num_of_sp]  # this is a constant horizon, if set to 'do not use', we'll use the HORIZON_MULTIPLIER
GAMMA = 0.9  # discount factor
assert GAMMA < 1, 'Gamma must be < 1 for non-finite horizon'

# exploration policy params
INIT_T_SOFTMAX = 1
FINAL_T_SOFTMAX = 1
EXPLORATION_POLICY = ExploreScheduler.SoftMaxSchedule(
    schedule_time_steps=EPISODES*HORIZON*0.75,
    final_t=FINAL_T_SOFTMAX,
    initial_t=INIT_T_SOFTMAX)

# target net params
USE_TARGET = False
TARGET_NET_UPDATE_POLICY = 'hard'
SOFT_UPDATE_RATE = 1e-3  # for soft update of target parameters
UPDATE_TARGET_NET = 100000


##########################
####### MAIN PATHS #######
##########################
DIRPATH = Path('./')
PATH_TO_RAW_TREE_DATA = DIRPATH / 'Tree_Data/'
EXPERIMENTS_RESDIR = DIRPATH / 'experiments_results'
RESULTS_FOLDER_NAME = 'results_and_graphs'
# better to divide to classes (e.g., AGENT (containing QNET and LL_ESTIMATOR) | FEATURE_EXTRACTOR | DATA | GENERAL) ?

##########################
### FEATURE EXTRACTOR ####
##########################
NJ_TBL_FOR_DATASET_DICT = NJ_TBL_FOR_DATASET_DICT
# note: for naming clarifications, refer to In_Progress_Code/FeatureExtractor/Visual_intuition_for_trees.png
TARGET_LABEL_COLUMN = 'd_ll'  # edited from "d_ll_merged" that was there for historical reasons only
CURRENT_TREE_NEWICK_COLUMN_NAME = 'current_tree_newick'
SPR_MOVE_PRUNE_BRANCH = 'prune_branch'
SPR_MOVE_RGRFT_BRANCH = 'rgrt_branch'
NBOOTREES = 200
LABEL_MULTIPLIER = 1000
DATA_SET_GROUPING_COLUMN_NAME = 'data_set_number'
METADATA_LIST = [SPR_MOVE_PRUNE_BRANCH, SPR_MOVE_RGRFT_BRANCH, DATA_SET_GROUPING_COLUMN_NAME]
LABEL_LIST = [TARGET_LABEL_COLUMN, 'resulting_ll', 'current_ll']
FEATURE_LIST = ['total_branch_length_nj_tree',
                'total_branch_length_current_tree', 'longest_branch_current_tree', 'branch_length_prune',
                'branch_length_rgft', 'topo_dist_from_pruned', 'branch_dist_from_pruned',
                'bstrap_nj_prune_current_tree', 'bstrap_nj_rgft_current_tree', 'bstrap_upgma_prune_current_tree',
                'bstrap_upgma_rgft_current_tree', 'number_of_species_b_subtree', 'total_branch_length_b_subtree',
                'longest_branch_b_subtree', 'number_of_species_c_subtree', 'total_branch_length_c_subtree',
                'longest_branch_c_subtree', 'number_of_species_prune', 'total_branch_length_prune',
                'longest_branch_prune', 'number_of_species_remaining', 'total_branch_length_remaining',
                'longest_branch_remaining', 'bstrap_nj_prune_resulting', 'bstrap_nj_rgft_resulting',
                'bstrap_upgma_prune_resulting', 'bstrap_upgma_rgft_resulting']

IN_FEATURES = len(FEATURE_LIST)
DATA_TYPES = {name: str if name in METADATA_LIST else float for name in
              FEATURE_LIST + LABEL_LIST + METADATA_LIST}

##########################
######### DATA ###########
##########################
RAXML_NG_SCRIPT = "raxml-ng"  # after you install raxml-ng on your machine
PATH_TO_SUDO_STARTING_TREE = Path('SudoStartingTree.phy')
PATH_TO_SUDO_STARTING_ACTION = Path('SudoStartingTreeMove.json')
# file names that need to be in each data folder:
MSA_PHYLIP_FILENAME_NOT_MASKED = "real_msa.phy"  # we deleted for all dirs the original msa file (Dana has it zipped together with the name_conversion_dict, if necessary
MSA_FILE_NAME = Path('masked_species_real_msa.phy')
SPLIT_HASH_NJ = Path('SplitsHash_nj.pkl')
SPLIT_HASH_UPGMA = Path('SplitsHash_upgma.pkl')
NJ_STARTING_TREE_FILE_NAME = Path('masked_species_real_msa.phy_phyml_tree_bionj.txt')
PHYML_PARAM_FILE_NAME = Path('masked_species_real_msa.phy_phyml_stats_bionj.txt')
RAXML_ML_TREE_FILE_NAME = Path("masked_species_real_msa.phy.raxml.bestTree")
RAXML_TEMP_ML_TREE_FILE_NAME_PREFIX = Path("TEMP_RAXML_FILE")
RAXML_TEMP_ML_TREE_FILE_NAME = Path("TEMP_RAXML_FILE.raxml.bestTree")
TEMP_TREE_FOR_RAXML_SEARCH_FILE_NAME = Path("Temp_start_tree_file_for_raxml_search.txt")
NEEDED_FILES_IN_A_FOLDER = [MSA_FILE_NAME, NJ_STARTING_TREE_FILE_NAME, SPLIT_HASH_NJ, SPLIT_HASH_UPGMA,
                            PHYML_PARAM_FILE_NAME, RAXML_ML_TREE_FILE_NAME]
RANDOM_STARTING_TREE_OPTIONS_FILE_NAME = Path('random_starting_trees_{}_trees_horizon_{}.txt')
PATH_TO_TESTING_TREES_FILE = Path("Tree_data/sampled_datasets_All_sized_ds_Test_RL.csv")
PATH_TO_TRAINING_TREES_FILE = Path("Tree_data/sampled_datasets_All_sized_ds_Train_RL.csv")

RANDOM_STARTING_TREES_FILE_NAME = "{data_set}.raxml.startTree"

##########################
######### AGENT ##########
##########################
USE_CUDA = torch.cuda.is_available()
Q_NETWORK_LOCAL_WEIGHTS_FILE_NAME = Path('Qnetwork_local_weights.pkl')
Q_NETWORK_TARGET_WEIGHTS_FILE_NAME = Path('Qnetwork_target_weights.pkl')
REPLAY_BUFFER_FILE_NAME = Path('Replay_Buffer_memory.pkl')
PROCESSED_DATA_FILE_NAME = Path('Processed_Data.pkl')
TESTING_RESULT_COLUMNS = ["test_replication_number", "data_set_number", "ll_diff", "rf_dist", "ll_norm_factor", "rf_norm_factor", "raxml_ml", "max_ll_reached"]


##########################
## LEARN FROM EXPERIENCE #
##########################

EPOCHS = 501
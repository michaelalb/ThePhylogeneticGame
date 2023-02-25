import json
import os
import pickle
import random
import time

import SharedConsts as SC
from FeatureExtractor.FeatureExtractor import FeatureExtractorClass
from FeatureExtractor.FeatureExtractorGeneralToolBox import extract_all_neibours_features_multiprocessing
from Reinforcement_env.env_utils import PhyloGameUtils
from SPR_generator.SPR_move import get_moves_from_obj, generate_base_neighbour, generate_tree_object, Edge


class PhyloGame:

	def __init__(self, number_of_cpus_to_use, results_dir, is_train, datasets=None,
				 use_random_starts=False):
		"""
		create file extractor object and initialize the spr environment
		one PhyloGame instance is to be created per training session
		"""
		self.horizon = None
		self.tree = None
		self.likelihood = None
		self.all_actions = None
		self.split_hash_dict = {}
		self.fe = FeatureExtractorClass()
		self.helper = PhyloGameUtils(feature_extractor_instance=self.fe, datasets=datasets,
									 use_random_starts=use_random_starts, results_dir=results_dir)
		self.number_of_cpus = number_of_cpus_to_use
		self.is_train = is_train
		self.reset()

	@staticmethod
	def get_state_action_size():
		return len(SC.FEATURE_LIST)

	def reset(self, dataset=None):
		"""
		reset the environment: choose new random data set to start from,
		reset the horizon count
		"""
		# tree is a tree object from SPR_generator
		if dataset:
			self.tree = self.helper.get_starting_tree(dataset)
		else:
			self.tree = self.helper.get_starting_tree()

		self.likelihood = self.fe.extract_likelihood(self.tree, self.helper.data_set)['resulting_ll'][0]
		self.horizon = self.helper.get_horizon(tree_obj=self.tree)
		if self.split_hash_dict.get(self.helper.data_set) is None:
			split_hash_dicts = {}
			splits_btsrap_hash_file_path = SC.PATH_TO_RAW_TREE_DATA / self.helper.data_set
			if not (splits_btsrap_hash_file_path / "SplitsHash_upgma.pkl").is_file():
				self.split_hash_dict[self.helper.data_set] = split_hash_dicts
				return
			with open(str(splits_btsrap_hash_file_path / "SplitsHash_upgma.pkl"), 'rb') as fp:
				split_hash_dicts['upgma'] = pickle.load(fp)
			with open(str(splits_btsrap_hash_file_path / "SplitsHash_nj.pkl"), 'rb') as fp:
				split_hash_dicts['nj'] = pickle.load(fp)
			self.split_hash_dict[self.helper.data_set] = split_hash_dicts
		return self.tree

	def get_all_possible_actions(self):
		self.all_actions = get_moves_from_obj(self.tree)
		return self.all_actions.copy()

	def get_all_neighbor_states(self):
		"""
		the agent calls this method to receive all the features relevant to evaluate at the current step.
		the feature-to-action dict is used to match the agent's choice to a specific action
		(the agent only knows what features are, not actions)
		"""
		# we need to save all actions to mem so we can take the correct action on step()
		self.all_actions = get_moves_from_obj(self.tree)
		feature_lst = extract_all_neibours_features_multiprocessing(current_tree_obj=self.tree,
																	tool_box_instance=self.fe,
																	all_moves=self.all_actions,
																	data_set_num=self.helper.data_set,
																	number_of_cpus=self.number_of_cpus,
																	calculation_flag='features',
																	result_format='vector',
																	normalization_factor=self.helper.get_normalization_factor_per_ds(data_set=self.get_data_set()),
																	split_hash_dict=self.split_hash_dict.get(self.helper.data_set))
		assert len(self.all_actions) == len(feature_lst)  # assert multiprocessing really returns all results every time

		return self.helper.transform(feature_lst)

	def get_random_neighbor_state(self):
		"""
		the agent calls this method to receive all the features relevant to evaluate at the current step.
		the feature-to-action dict is used to match the agent's choice to a specific action
		(the agent only knows what features are, not actions)
		"""
		# we need to save all actions to mem so we can take the correct action on step()
		self.all_actions = get_moves_from_obj(self.tree)
		random_index = random.randint(0, len(self.all_actions) - 1)
		action = [self.all_actions[random_index]]
		feature_lst = extract_all_neibours_features_multiprocessing(current_tree_obj=self.tree,
																	tool_box_instance=self.fe,
																	all_moves=action,
																	data_set_num=self.helper.data_set,
																	number_of_cpus=self.number_of_cpus,
																	calculation_flag='features',
																	result_format='vector',
																	normalization_factor=self.helper.get_normalization_factor_per_ds(data_set=self.get_data_set()),
																	split_hash_dict=self.split_hash_dict.get(self.helper.data_set))

		return self.helper.transform(feature_lst), random_index

	def get_action(self, index):
		return self.all_actions[index]

	def step(self, state_action_features_index, should_calculate_reward=True):
		"""
		make one spr move, use agent to evaluate actions,
		notice this makes the environment in charge of the policy, so will change pretty soon,
		the agent will receive a list of features and return the chosen features (index)
		according to its policy
		"""
		action = self.get_action(state_action_features_index)
		self.tree = generate_base_neighbour(self.tree, action)
		if should_calculate_reward:
			res = self.fe.extract_likelihood(
				self.tree, self.helper.data_set, previous_ll=self.likelihood,
				normalization_factor=self.helper.get_normalization_factor_per_ds(data_set=self.get_data_set()))
			self.likelihood = res['resulting_ll'][0]
			if res['resulting_ll'][1] is not None:  # this means: SHOULD_OPTIMIZE_BRANCH_LENGTHS = False
				self.tree = res['resulting_ll'][1]
			reward = res[SC.TARGET_LABEL_COLUMN]
		else:
			reward = 0
		done = self.step_horizon()

		return reward, done

	def step_horizon(self):
		"""
		keep track of horizon
		"""
		self.horizon -= 1
		if self.horizon <= 0:
			return True
		return False

	def get_data_set(self):
		return self.helper.data_set

	def parse_move(self, move_json):
		prune = (Edge(node_a=move_json['prune_a'], node_b=move_json['prune_b']))
		rgft = (Edge(node_a=move_json['rgft_a'], node_b=move_json['rgft_b']))
		return (prune, rgft)

	def generate_false_start(self):
		"""
		this function is created to mitigate the fact the we cannot commit to memory our first action
		because we use state-action features so for the first move there is no "old state-action" prior to the move
		** to make sure this is always constant for each data set we make sure the neighbour selection is deterministic
		and we always take the first option
		:return: "old state action features"
		"""
		sudo_tree_obj = generate_tree_object(SC.PATH_TO_RAW_TREE_DATA / self.helper.data_set / SC.PATH_TO_SUDO_STARTING_TREE)
		with open(SC.PATH_TO_RAW_TREE_DATA / self.helper.data_set / SC.PATH_TO_SUDO_STARTING_ACTION, 'r') as fp:
			sudo_move_json = json.load(fp)
		sudo_first_move = self.parse_move(sudo_move_json)
		sudo_res = self.fe.extract_features(sudo_tree_obj, data_set_number=self.helper.data_set, move=sudo_first_move,
											result_format='vector', calculation_flag='features')
		return self.helper.transform([sudo_res])

	def get_raxml_likelihood(self, how_many_starting_trees=20):
		tree_suffix = str(SC.RAXML_ML_TREE_FILE_NAME) + f'_{how_many_starting_trees}_only_random'
		start_tree_path = SC.PATH_TO_RAW_TREE_DATA / self.helper.data_set / tree_suffix
		if not start_tree_path.is_file():
			self.helper.generate_raxml_max_liklihood_tree(data_set=self.helper.data_set,
														  total_trees=how_many_starting_trees)
			tmp_tree_path = SC.PATH_TO_RAW_TREE_DATA / self.helper.data_set / SC.RAXML_ML_TREE_FILE_NAME
			tmp_tree_path.replace(start_tree_path)
		tree_obj = generate_tree_object(start_tree_path)
		result = self.fe.extract_likelihood(current_tree_obj=tree_obj, data_set_number=self.helper.data_set)

		return result['current_ll'][0]

	def get_raxml_likelihood_specific_starting_tree(self, tree_object):
		temp_tree_suffix = str(SC.TEMP_TREE_FOR_RAXML_SEARCH_FILE_NAME)
		start_tree_path = SC.PATH_TO_RAW_TREE_DATA / self.helper.data_set / temp_tree_suffix
		with open(start_tree_path, 'w+') as fp:
			fp.write(str(tree_object.write(format=1)))
		self.helper.generate_raxml_max_liklihood_tree_from_specific_start(data_set=self.helper.data_set,
																		  start_tree_path=start_tree_path)
		resulting_tree_path = SC.PATH_TO_RAW_TREE_DATA / self.helper.data_set / SC.RAXML_ML_TREE_FILE_NAME
		resulting_tree_path = resulting_tree_path.replace(str(resulting_tree_path) + f'_{time.time()}.txt')
		tree_obj = generate_tree_object(resulting_tree_path)
		result = self.fe.extract_likelihood(current_tree_obj=tree_obj, data_set_number=self.helper.data_set)
		os.remove(start_tree_path)
		os.remove(resulting_tree_path)
		return result['current_ll'][0]

	def get_NJ_likelihood(self, data_set):
		NJ_tree_path = SC.PATH_TO_RAW_TREE_DATA / data_set / SC.NJ_STARTING_TREE_FILE_NAME
		tree_obj = generate_tree_object(NJ_tree_path)
		result = self.fe.extract_likelihood(current_tree_obj=tree_obj, data_set_number=data_set)
		return result['current_ll'][0]

	def remove_files(self):
		self.helper.remove_files()


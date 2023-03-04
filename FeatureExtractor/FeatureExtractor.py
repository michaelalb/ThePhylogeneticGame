import numpy as np
import pandas as pd

import FeatureExtractor.FeatureExtractorFeatureToolBox as FeatureToolBox
import SPR_generator.SPR_move as SPR
import SharedConsts as SC


class FeatureExtractorClass:
    """
    orchestrator class for feature calculations and SPR steps.
    should be a helper class for the reinforcement learning agent
    please use the visual intuition - it corresponds to variable names
    """

    def __init__(self):
        self.FTB = None

    def order_data(self, current_tree_string, prune_node, rgrt_node, results_dict, calculation_flag, normalization_factor,
                   data_set_number=None, result_format='dict', return_tree_string=False, should_transform_results=True):
        """
        this function organizes the data in a pandas DF.
        :param current_tree_string: the newick formatted tree of the current tree (pre SPR)
        :param prune_node: the node ABOVE the branch we pruned at
        :param rgrt_node: the node ABOVE the branch we grafted at
        :param results_dict: a dictionary with the results {feature_name:score}
        :param data_set_number: number id of the dataset this tree came from (datasets are arranged by starting tree
        in the training data
        :param return_tree_string: flag to determine whether tree string should be in the res
        :param result_format: 'dict'\'episode_logger'\'vector' (vector is only the values)
        :param should_transform_results : flag to determine whether results should be normalized
        :param normalization_factor: has to be positive because ll's are all negative
        :return: a pandas episode_logger with the results and metadata
        """
        unique_id = 0
        results = results_dict
        # add naming features
        if return_tree_string:
            results[SC.CURRENT_TREE_NEWICK_COLUMN_NAME] = current_tree_string
        results[SC.SPR_MOVE_PRUNE_BRANCH] = prune_node
        results[SC.SPR_MOVE_RGRFT_BRANCH] = rgrt_node
        # this is meant to map result to dataset number! taken from training data folders.
        # this + prune name + rgft_name will be used as a starting tree and its neibours unique id.
        results[SC.DATA_SET_GROUPING_COLUMN_NAME] = data_set_number
        if calculation_flag != 'features':
            if normalization_factor < 0: # just a note: the returned ll is always the -log(likelihood) so this condition is always true. We could just take the abs/minus as the normalization factor in the first place.
                normalization_factor *= -1
            results[SC.TARGET_LABEL_COLUMN] = ((results['resulting_ll'][0] - results['current_ll'][0]) / normalization_factor) * SC.LABEL_MULTIPLIER
            assert results['current_ll'][0] < 0, results['current_ll'][0]  # the normalization above is dependent on the fact
            # that the current is always negative
        else:
            results['total_branch_length_nj_tree'] = SC.NJ_TBL_FOR_DATASET_DICT[data_set_number]


        if result_format == 'episode_logger':
            results = pd.DataFrame.from_dict({unique_id: results_dict}, orient='index')
        elif result_format == 'vector':
            only_feat = [results[x] for x in SC.FEATURE_LIST]
            results = np.array(only_feat).reshape(len(only_feat), 1)
        return results

    def calculate_tree_features(self, which_tree, prune_edge, rgft_edge, calculation_flag):
        """
        this is the worker function. iterating over the relevant tree and its associated functions.
        the result dictionary that gets created from the associated dict.
        assumes the ToolBox class has a dictionary for each tree type which is of the structure
        {feature_name: feature_func}
        :param which_tree: the KIND of tree we are calculating on. the options are:
        'current_tree', 'prune_tree', 'remaining_tree','b_subtree', 'c_subtree', 'resulting_tree'
        :param prune_edge: the edge to be pruned
        :param rgft_edge: the node to be re-grafted
        :param calculation_flag: this is a string that flags the function what to calculate: (case insensitive)
        all: features and likelihood
        ll: just the likelihood
        features: just the features
        :return: the relevant trees results
        """
        calculation_flag = calculation_flag.lower()
        if calculation_flag == 'all':
            tree_results = {name: 0 for name in self.FTB.tree_mappings[which_tree].keys()}
        elif calculation_flag == 'll':
            tree_results = {name: 0 for name in self.FTB.tree_mappings[which_tree].keys() if name.find('ll') != -1}
        elif calculation_flag == 'features':
            tree_results = {name: 0 for name in self.FTB.tree_mappings[which_tree].keys() if name.find('ll') == -1}
        for name in tree_results.keys():
            func = self.FTB.tree_mappings[which_tree][name]
            tree_results.update({name: func(which_tree=which_tree, prune_edge=prune_edge, rgft_edge=rgft_edge)})
        return tree_results

    def extract_features(self, current_tree_obj, move, data_set_number=None, should_transform_results=True, queue=None,
                         calculation_flag='all', result_format='dict', return_tree_string=False, normalization_factor=1,
                         split_hash_dict=None):
        """
        this is the orchastrator function - receiving a tree , preforming the relevant SPR move and calculating the
        relevant features for each tree/subtree.
        this is achieved using the FeatureExtractorFeatureToolBox class where each of the subtrees we want to
        calculate features for has a dictionary of features associated with it.
        :param current_tree_obj: current tree ete object
        :param move: a tuple of a possible move containing- the branch to be pruned (we prune ABOVE branch.node_a), and the branch to be regrafted (we regraft ABOVE branch.node_a)
        :param data_set_number: number id of the dataset this tree came from (datasets are arranged by starting tree)
        :param should_transform_results: flag to determine whether results should be normalized
        :param queue: a multiprocessing queue
        :param result_format: 'dict'\'episode_logger'\'vector' (vector is only the values)
        :param calculation_flag: this is a string that flags the function what to calculate: (case insensitive)
        :param normalization_factor: has to be negative because ll's are all negative
        all: features and likelihood
        ll: just the likelihood
        features: just the features
        :return: a single lined dataframe with all the starting tree, spr move an feature data.
        feature names can be found in the ToolBox class.
        """
        resulting_tree, prune_subtree, remaining_tree, b_subtree, c_subtree = SPR.generate_neighbour(base_tree=current_tree_obj, possible_move=move)
        prune_edge, rgft_edge = move

        self.FTB = FeatureToolBox.FeatureExtractorFeatureToolBoxClass(current_tree=current_tree_obj,
                                                                      prune_tree=prune_subtree,
                                                                      remaining_tree=remaining_tree,
                                                                      b_subtree=b_subtree,
                                                                      c_subtree=c_subtree,
                                                                      data_set_number=data_set_number,
                                                                      resulting_tree=resulting_tree,
                                                                      split_hash_dict=split_hash_dict)
        current_tree_results = self.calculate_tree_features(which_tree='current_tree', prune_edge=prune_edge,
                                                            rgft_edge=rgft_edge, calculation_flag=calculation_flag)
        prune_tree_results = self.calculate_tree_features(which_tree='prune_tree', prune_edge=prune_edge,
                                                          rgft_edge=rgft_edge, calculation_flag=calculation_flag)
        remaining_tree_results = self.calculate_tree_features(which_tree='remaining_tree', prune_edge=prune_edge,
                                                              rgft_edge=rgft_edge, calculation_flag=calculation_flag)
        b_subtree_tree_results = self.calculate_tree_features(which_tree='b_subtree', prune_edge=prune_edge,
                                                              rgft_edge=rgft_edge, calculation_flag=calculation_flag)
        c_subtree_tree_results = self.calculate_tree_features(which_tree='c_subtree', prune_edge=prune_edge,
                                                              rgft_edge=rgft_edge, calculation_flag=calculation_flag)
        resulting_tree_results = self.calculate_tree_features(which_tree='resulting_tree', prune_edge=prune_edge,
                                                              rgft_edge=rgft_edge, calculation_flag=calculation_flag)

        combined_results = {**current_tree_results, **prune_tree_results, **remaining_tree_results,
                            **b_subtree_tree_results, **c_subtree_tree_results, **resulting_tree_results}

        tree_str = current_tree_obj.write(format=1) if return_tree_string else ''
        results = self.order_data(tree_str, prune_edge.node_a, rgft_edge.node_a,
                                  combined_results, data_set_number=data_set_number, calculation_flag=calculation_flag,
                                  result_format=result_format, return_tree_string=return_tree_string,
                                  should_transform_results=should_transform_results,
                                  normalization_factor=normalization_factor)

        if queue is None:
            return results
        else:
            queue.put(results)

    def extract_likelihood(self, current_tree_obj, data_set_number, queue=None, previous_ll=None, normalization_factor=1):
        """
        this function is used to calculate the likelihood of ONE specific tree, not to be confused with the use of
        extract_features with the 'll' flag which is used to get only the ll of a tree, make a spr move and get the ll
        of the resulting tree as well.
        :param current_tree_obj: tree we wish to get the ll for
        :param data_set_number: the data set number from which the tree arrives
        :param queue: a multiprocessing queue
        :param previous_ll: this is in case this function is being used by the env - in that case we want the result to
        be normalized so because this process happens here, we pass the previous ll
        :return: ll
        """
        self.FTB = FeatureToolBox.FeatureExtractorFeatureToolBoxClass(current_tree=current_tree_obj, prune_tree=None,
                                                                      remaining_tree=None, b_subtree=None,
                                                                      c_subtree=None, data_set_number=data_set_number,
                                                                      resulting_tree=None,
                                                                      split_hash_dict={})
        current_tree_results = self.calculate_tree_features(which_tree='current_tree', prune_edge=None,
                                                            rgft_edge=None, calculation_flag='ll')

        current_tree_results['resulting_ll'] = current_tree_results['current_ll']
        if previous_ll is not None:
            current_tree_results['current_ll'] = [previous_ll]
        results = self.order_data(current_tree_obj.write(format=1), None, None, current_tree_results,
                                  data_set_number=data_set_number, calculation_flag='ll', normalization_factor=normalization_factor)
        return results





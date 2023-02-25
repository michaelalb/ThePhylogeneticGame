import itertools
import os
import pickle

import numpy as np

import FeatureExtractor.FeatureExtractorGeneralToolBox as FGT
import SharedConsts as SC


class FeatureExtractorFeatureToolBoxClass:
    """
    this class houses the feature calculation methods.
    each type of subtree we want to calculate must be mentioned in the trees attribute
    AND in the tree mappings attribute with its associated calculation functions
    any data that need to be stored can be stored in tree_helper_param attribute
    ALSO this module is written to be multistringed - please have that in mind when you add - for example
    if you have two functions using the same data, but the first one need to calculate it - add an if statement to make
    is order insensitive - EXAMPLE: feature_topo_dist_from_pruned and feature_branch_dist_from_pruned
    all feature methods should start with feature prefix , all helper methods should be private
    """
    def __init__(self, current_tree, prune_tree, remaining_tree, b_subtree, c_subtree, resulting_tree, data_set_number,
                 split_hash_dict):
        self.data_set_number = data_set_number
        # trees
        self.trees = {'current_tree': current_tree, 'prune_tree': prune_tree, 'remaining_tree': remaining_tree,
                      'b_subtree': b_subtree, 'c_subtree': c_subtree, 'resulting_tree': resulting_tree}
        # function mappings

        self.tree_mappings = {
            'current_tree': {'total_branch_length_current_tree': self.feature_tbl,
                             'longest_branch_current_tree': self.feature_longest_branch,
                             'branch_length_prune': self.feature_branch_length_prune,
                             'branch_length_rgft': self.feature_branch_length_rgft,
                             'topo_dist_from_pruned': self.feature_topo_dist_from_pruned,
                             'branch_dist_from_pruned': self.feature_branch_dist_from_pruned,
                             'bstrap_nj_prune_current_tree': self.features_bootstrap_nj_prune,
                             'bstrap_nj_rgft_current_tree': self.features_bootstrap_nj_rgft,
                             'bstrap_upgma_prune_current_tree': self.features_bootstrap_upgma_prune,
                             'bstrap_upgma_rgft_current_tree': self.features_bootstrap_upgma_rgft,
                             'current_ll': self.feature_likelihood
                             },
            'b_subtree': {'number_of_species_b_subtree': self.feature_number_of_species,
                          'total_branch_length_b_subtree': self.feature_tbl,
                          'longest_branch_b_subtree': self.feature_longest_branch,
                          },
            'c_subtree': {'number_of_species_c_subtree': self.feature_number_of_species,
                          'total_branch_length_c_subtree': self.feature_tbl,
                          'longest_branch_c_subtree': self.feature_longest_branch,
                          },
            'prune_tree': {'number_of_species_prune': self.feature_number_of_species,
                           'total_branch_length_prune': self.feature_tbl,
                           'longest_branch_prune': self.feature_longest_branch,
                           },
            'remaining_tree': {'number_of_species_remaining': self.feature_number_of_species,
                               'total_branch_length_remaining': self.feature_tbl,
                               'longest_branch_remaining': self.feature_longest_branch,
                               },
            'resulting_tree': {'bstrap_nj_prune_resulting': self.features_bootstrap_nj_prune,
                               'bstrap_nj_rgft_resulting': self.features_bootstrap_nj_rgft,
                               'bstrap_upgma_prune_resulting': self.features_bootstrap_upgma_prune,
                               'bstrap_upgma_rgft_resulting': self.features_bootstrap_upgma_rgft,
                               'resulting_ll': self.feature_likelihood
                                   }
            }
        # params
        self.tree_helper_params = {
            'current_tree': {'branch_lengths': None, 'distance_from_pruned': None, 'BL_Dana': None, 'stemminess_indexes_Dana': None, 'frac_of_cherries_Dana': None, 'PBN_Dana': None, 'PBL_Dana': None},
            'prune_tree': {'ntaxa_tree': None, 'ntaxa_prune': None, 'branch_lengths': None, 'BL_Dana': None, 'stemminess_indexes_Dana': None, 'frac_of_cherries_Dana': None, 'PBN_Dana': None, 'PBL_Dana': None},
            'remaining_tree': {'ntaxa_tree': None, 'ntaxa_prune': None, 'branch_lengths': None, 'BL_Dana': None, 'stemminess_indexes_Dana': None, 'frac_of_cherries_Dana': None, 'PBN_Dana': None, 'PBL_Dana': None},
            'b_subtree': {'ntaxa_tree': None, 'ntaxa_prune': None, 'branch_lengths': None, 'BL_Dana': None, 'stemminess_indexes_Dana': None, 'frac_of_cherries_Dana': None, 'PBN_Dana': None, 'PBL_Dana': None},
            'c_subtree': {'ntaxa_tree': None, 'ntaxa_prune': None, 'branch_lengths': None, 'BL_Dana': None, 'stemminess_indexes_Dana': None, 'frac_of_cherries_Dana': None, 'PBN_Dana': None, 'PBL_Dana': None},
            'resulting_tree': {'ntaxa_tree': None, 'ntaxa_prune': None, 'branch_lengths': None, 'BL_Dana': None,
                          'stemminess_indexes_Dana': None, 'frac_of_cherries_Dana': None, 'PBN_Dana': None,
                          'PBL_Dana': None},

            'shared': {'SplitsHash': {'nj': split_hash_dict.get('nj'), 'upgma': split_hash_dict.get('upgma')}}
        }

    def get_all_feature_names(self):
        all_names = []
        for tree in self.tree_mappings.keys():
            for name in self.tree_mappings[tree].keys():
                all_names.append(name)
        return all_names

    @staticmethod
    def __get_branch_lengths(tree, is_prune_tree=False):
        """
        :param tree: Tree node or tree file or newick tree string;
        :return: list of branch lengths
        """
        tree_root = tree.get_tree_root()
        if len(tree) == 1 and not "(" in tree.write(format=1):  # in one-branch trees, sometimes the newick string is without "(" and ")" so the .iter_decendants returns None
            return [tree.dist]
        branches = []
        for node in tree_root.iter_descendants():
            branches.append(node.dist)
        if is_prune_tree:
            branches.append(tree_root.dist)
        return branches

    @staticmethod
    def __dist_between_nodes(tree, prune_node, rgft_node):
        nleaves_between = prune_node.get_distance(rgft_node,
                                                  topology_only=True) + 1  # +1 to convert between nodes count to edges
        tbl_between = prune_node.get_distance(rgft_node, topology_only=False)
        return {'nleaves_between': nleaves_between, 'tbl_between': tbl_between}

    def feature_tbl(self, **kwargs):
        if self.tree_helper_params[kwargs['which_tree']]['branch_lengths'] is None:
            is_prune_tree = True if kwargs['which_tree'] == 'prune_tree' else False
            branches = self.__get_branch_lengths(tree=self.trees[kwargs['which_tree']], is_prune_tree=is_prune_tree)
            self.tree_helper_params[kwargs['which_tree']]['branch_lengths'] = branches
            return sum(branches)
        else:
            return sum(self.tree_helper_params[kwargs['which_tree']]['branch_lengths'])

    def feature_longest_branch(self, **kwargs):
        if self.tree_helper_params[kwargs['which_tree']]['branch_lengths'] is None:
            is_prune_tree = True if kwargs['which_tree'] == 'prune_tree' else False
            branches = self.__get_branch_lengths(tree=self.trees[kwargs['which_tree']], is_prune_tree=is_prune_tree)
            self.tree_helper_params[kwargs['which_tree']]['branch_lengths'] = branches
            return max(branches)
        else:
            return max(self.tree_helper_params[kwargs['which_tree']]['branch_lengths'])

    def feature_branch_length_prune(self, **kwargs):
        node = (self.trees['current_tree'] & kwargs['prune_edge'].node_a)
        return node.dist

    def feature_branch_length_rgft(self, **kwargs):
        node = (self.trees[kwargs['which_tree']] & kwargs['rgft_edge'].node_a)
        return node.dist

    def feature_topo_dist_from_pruned(self, **kwargs):
        prunde_node = self.trees[kwargs['which_tree']] & kwargs['prune_edge'].node_a
        rgft_node = self.trees[kwargs['which_tree']] & kwargs['rgft_edge'].node_a
        if self.tree_helper_params[kwargs['which_tree']]['distance_from_pruned'] is None:
            results = self.__dist_between_nodes(tree=self.trees[kwargs['which_tree']], prune_node=prunde_node,
                                                rgft_node=rgft_node)
            self.tree_helper_params[kwargs['which_tree']]['distance_from_pruned'] = results
            return results['nleaves_between']
        else:
            return self.tree_helper_params[kwargs['which_tree']]['distance_from_pruned']['nleaves_between']

    def feature_branch_dist_from_pruned(self, **kwargs):
        prune_node = self.trees[kwargs['which_tree']] & kwargs['prune_edge'].node_a
        rgft_node = self.trees[kwargs['which_tree']] & kwargs['rgft_edge'].node_a
        if self.tree_helper_params[kwargs['which_tree']]['distance_from_pruned'] is None:
            results = self.__dist_between_nodes(tree=self.trees[kwargs['which_tree']], prune_node=prune_node,
                                                rgft_node=rgft_node)
            self.tree_helper_params[kwargs['which_tree']]['distance_from_pruned'] = results
            return results['tbl_between']
        else:
            return self.tree_helper_params[kwargs['which_tree']]['distance_from_pruned']['tbl_between']

    def feature_number_of_species(self, **kwargs):
        tree = self.trees['current_tree'].get_tree_root()
        intersecting_tree = self.trees[kwargs['which_tree']].get_tree_root()
        if self.tree_helper_params[kwargs['which_tree']]['ntaxa_tree'] is None:
            ntaxa_tree = len(tree)
            ntaxa_intersecting = len(intersecting_tree)
            self.tree_helper_params[kwargs['which_tree']]['ntaxa_tree'] = ntaxa_tree
            self.tree_helper_params[kwargs['which_tree']]['ntaxa_prune'] = ntaxa_intersecting
            return ntaxa_intersecting
        else:
            return self.tree_helper_params[kwargs['which_tree']]['ntaxa_prune']

    def features_bootstrap_nj_prune(self, **kwargs):
        branch_name = kwargs['prune_edge'].node_a
        if kwargs['which_tree'] == 'resulting_tree':
            one_up_name = kwargs['prune_edge'].node_b
            starting_tree = self.trees['current_tree']
            # if regular case we want to locate the up_node name of the original tree in the resulting tree.
            # if not regular case (namely a child of the ROOT):
            # (1) if branch leads to leaf- ROOT. OR (2) branch leads to full clades- one_up_name
            branch_name = ((starting_tree & one_up_name).up).name if not one_up_name == "ROOT" else "ROOT" if "Sp" in kwargs['prune_edge'].node_a else one_up_name
        return self.features_bootstrap(which_tree=kwargs['which_tree'],
                                       bstrap_algo='nj', branch_name=branch_name)

    def features_bootstrap_nj_rgft(self, **kwargs):
        branch_name = kwargs['rgft_edge'].node_a if kwargs['which_tree'] != 'resulting_tree' else kwargs['prune_edge'].node_b
        return self.features_bootstrap(which_tree=kwargs['which_tree'],
                                       bstrap_algo='nj', branch_name=branch_name)

    def features_bootstrap_upgma_prune(self, **kwargs):
        branch_name = kwargs['prune_edge'].node_a
        if kwargs['which_tree'] == 'resulting_tree':
            one_up_name = kwargs['prune_edge'].node_b
            starting_tree = self.trees['current_tree']
            # if regular case we want to locate the up_node name of the original tree in the resulting tree.
            # if not regular case (namely a child of the ROOT):
            # (1) if branch leads to leaf- ROOT. OR (2) branch leads to full clades- one_up_name
            branch_name = ((starting_tree & one_up_name).up).name if not one_up_name == "ROOT" else "ROOT" if "Sp" in kwargs['prune_edge'].node_a else one_up_name
        return self.features_bootstrap(which_tree=kwargs['which_tree'],
                                       bstrap_algo='upgma', branch_name=branch_name)

    def features_bootstrap_upgma_rgft(self, **kwargs):
        branch_name = kwargs['rgft_edge'].node_a if kwargs['which_tree'] != 'resulting_tree' else kwargs['prune_edge'].node_b
        return self.features_bootstrap(which_tree=kwargs['which_tree'],
                                       bstrap_algo='upgma', branch_name=branch_name)

    def features_bootstrap(self, which_tree, bstrap_algo, branch_name):
        # todo: there is a bug still - upgma of current tree is sometimes 2 !?? for a certain node
        """
        bootstrap feature assignes a bootstrap value to a given edge in the tree.
        in order to have these values we create a hashing dictionary of all optional splits in species group.
        this data is saved to a hash table using the min group of each split  in lexicographic order
        example: 123|4567 will be saved as 123.
        if the hashing doesn't already exist we create it.
        """
        tree = self.trees[which_tree].get_tree_root()
        if self.tree_helper_params['shared']['SplitsHash'][bstrap_algo] is None:
            # create hash dict
            splits_btsrap_hash_file_path = SC.PATH_TO_RAW_TREE_DATA / self.data_set_number / "SplitsHash_{}.pkl".format(bstrap_algo)
            if os.path.exists(splits_btsrap_hash_file_path):
                with open(splits_btsrap_hash_file_path, "rb") as dict_file:
                    splitsHash = pickle.load(dict_file)
                    dict_file.close()
            else:
                btsrp_newicks_lst = FGT.generate_bootstrap_trees(data_set_number=self.data_set_number, algo=bstrap_algo, nbootrees=SC.NBOOTREES)
                species_lst = FGT.msa_to_species_lst(data_set_number=self.data_set_number)
                splitsHash = {}
                splitsHash = FGT.update_splitsHash(original_splitsHash=splitsHash, btsrp_newicks_lst=btsrp_newicks_lst, species_lst=species_lst)
                with open(splits_btsrap_hash_file_path, "wb") as dict_file:
                    pickle.dump(splitsHash, dict_file)
                    dict_file.close()
            self.tree_helper_params['shared']['SplitsHash'][bstrap_algo] = splitsHash
        else:
            splitsHash = self.tree_helper_params['shared']['SplitsHash'][bstrap_algo]
        #todo: this is due to the naming crisis. once resolved - remove.
        try:
            bstrap_val = FGT.split_lookup(splitsHash=splitsHash, anyNode=(tree&branch_name), data_set_number=self.data_set_number)
        except:
            bstrap_val = 0
        return bstrap_val

    def feature_likelihood(self, **kwargs):
        msa_file_path = str(SC.PATH_TO_RAW_TREE_DATA / self.data_set_number / SC.MSA_FILE_NAME)
        stat_path = str(SC.PATH_TO_RAW_TREE_DATA / self.data_set_number / SC.PHYML_PARAM_FILE_NAME)
        freq, rates, pinv, alpha = FGT.get_likelihood_params(stat_path)
        return FGT.calc_likelihood(self.trees[kwargs['which_tree']].write(format=1, format_root_node=True), msa_file_path, rates, pinv, alpha, freq)

    def likelihood_ml(self, **kwargs):
        msa_file_path = str(SC.PATH_TO_RAW_TREE_DATA / self.data_set_number / SC.MSA_FILE_NAME)
        stat_path = str(SC.PATH_TO_RAW_TREE_DATA / self.data_set_number / SC.PHYML_PARAM_FILE_NAME)
        freq, rates, pinv, alpha = FGT.get_likelihood_params(stat_path)
        return FGT.calc_likelihood(self.trees[kwargs['which_tree']].write(format=1, format_root_node=True), msa_file_path, rates, pinv, alpha, freq)

    # todo: implement like a normal person if this works

    def features_bl_dana(self, **kwargs):
        if self.tree_helper_params[kwargs['which_tree']]['BL_Dana'] is None:
            bl_dana = self.calc_bl_dana(self.trees[kwargs['which_tree']])
            self.tree_helper_params[kwargs['which_tree']]['BL_Dana'] = bl_dana
            return bl_dana
        else:
            return self.tree_helper_params[kwargs['which_tree']]['BL_Dana']

    def calc_bl_dana(self, tree):
        branches = self.get_branch_lengths_dana(tree)
        if not branches:
            return 0,0,0,0,0
        entropy = self.compute_entropy_dana(branches)
        # returning max here although already computed separately in our orig set of features. # todo: remove the max(branches) from the original implementation of get_branch_lengths or rename this function
        return max(branches), min(branches), np.mean(branches), np.std(branches), entropy

    def get_branch_lengths_dana(self, tree):
        """
        :param tree: Tree node or tree file or newick tree string;
        :return: total branch lengths
        """
        # TBL
        tree_root = tree.get_tree_root()
        branches = []
        for node in tree_root.iter_descendants():  # the root dist is 1.0, we don't want it
            branches.append(node.dist)
        return branches

    def compute_entropy_dana(self, lst, epsilon=0.000001):
        if np.sum(lst) != 0:
            lst_norm = np.array(lst)/np.sum(lst)
        else:
            lst_norm = np.array(lst) + epsilon
        entropy = -1*sum(np.log2(lst_norm)*lst_norm)
        if np.isnan(entropy):
            lst_norm += epsilon
            entropy = -1*sum(np.log2(lst_norm)*lst_norm)
        return entropy

    # add five feature funcs
    def calc_bl_dana_max(self, **kwargs):
        return self.features_bl_dana(**kwargs)[0]

    def calc_bl_dana_min(self, **kwargs):
        return self.features_bl_dana(**kwargs)[1]

    def calc_bl_dana_mean(self, **kwargs):
        return self.features_bl_dana(**kwargs)[2]

    def calc_bl_dana_std(self, **kwargs):
        return self.features_bl_dana(**kwargs)[3]

    def calc_bl_dana_entropy(self, **kwargs):
        return self.features_bl_dana(**kwargs)[4]

    def features_pbl_dana(self, **kwargs):
        if self.tree_helper_params[kwargs['which_tree']]['PBL_Dana'] is None:
            pbl_dana = self.calc_pbl_dana(self.trees[kwargs['which_tree']])
            self.tree_helper_params[kwargs['which_tree']]['PBL_Dana'] = pbl_dana
            return pbl_dana
        else:
            return self.tree_helper_params[kwargs['which_tree']]['PBL_Dana']

    def calc_pbl_dana(self, tree):
        tree_root = tree.copy().get_tree_root()
        for node in tree_root.iter_descendants():
            node.dist = 1.0
        tree_diams = []
        leaves = list(tree_root.iter_leaves())
        for leaf1, leaf2 in itertools.combinations(leaves, 2):
            tree_diams.append(leaf1.get_distance(leaf2))
        entropy = self.compute_entropy_dana(tree_diams)

        if not tree_diams:
            return 0,0,0,0,0

        return max(tree_diams), min(tree_diams), np.mean(tree_diams), np.std(tree_diams), entropy

    # add five feature funcs
    def calc_pbl_dana_max(self, **kwargs):
        return self.features_pbl_dana(**kwargs)[0]

    def calc_pbl_dana_min(self, **kwargs):
        return self.features_pbl_dana(**kwargs)[1]

    def calc_pbl_dana_mean(self, **kwargs):
        return self.features_pbl_dana(**kwargs)[2]

    def calc_pbl_dana_std(self, **kwargs):
        return self.features_pbl_dana(**kwargs)[3]

    def calc_pbl_dana_entropy(self, **kwargs):
        return self.features_pbl_dana(**kwargs)[4]

    def features_pbn_dana(self, **kwargs):
        if self.tree_helper_params[kwargs['which_tree']]['PBN_Dana'] is None:
            pbn_dana = self.calc_pbn_dana(self.trees[kwargs['which_tree']])
            self.tree_helper_params[kwargs['which_tree']]['PBN_Dana'] = pbn_dana
            return pbn_dana
        else:
            return self.tree_helper_params[kwargs['which_tree']]['PBN_Dana']

    def calc_pbn_dana(self, tree):
        tree_root = tree.get_tree_root()
        tree_diams = []
        leaves = list(tree_root.iter_leaves())
        for leaf1, leaf2 in itertools.combinations(leaves, 2):
            tree_diams.append(leaf1.get_distance(leaf2))
        entropy = self.compute_entropy_dana(tree_diams)
        if not tree_diams:
            return 0,0,0,0,0
        return max(tree_diams), min(tree_diams), np.mean(tree_diams), np.std(tree_diams), entropy

    # add five feature funcs
    def calc_pbn_dana_max(self, **kwargs):
        return self.features_pbn_dana(**kwargs)[0]

    def calc_pbn_dana_min(self, **kwargs):
        return self.features_pbn_dana(**kwargs)[1]

    def calc_pbn_dana_mean(self, **kwargs):
        return self.features_pbn_dana(**kwargs)[2]

    def calc_pbn_dana_std(self, **kwargs):
        return self.features_pbn_dana(**kwargs)[3]

    def calc_pbn_dana_entropy(self, **kwargs):
        return self.features_pbn_dana(**kwargs)[4]

    def features_frac_of_cherries_dana(self, **kwargs):
        if self.tree_helper_params[kwargs['which_tree']]['frac_of_cherries_Dana'] is None:
            frac_of_cherries = self.calc_frac_of_cherries_dana(self.trees[kwargs['which_tree']])
            self.tree_helper_params[kwargs['which_tree']]['frac_of_cherries_Dana'] = frac_of_cherries
            return frac_of_cherries
        else:
            return self.tree_helper_params[kwargs['which_tree']]['frac_of_cherries_Dana']

    def calc_frac_of_cherries_dana(self, tree):
        tree_root = tree.get_tree_root()
        leaves = list(tree_root.iter_leaves())
        cherries_cnt = 0
        for leaf1, leaf2 in itertools.combinations(leaves, 2):
            if leaf1.up is leaf2.up:
                cherries_cnt += 1
        return 2 * cherries_cnt / len(leaves)


    def features_stemminess_indexes_dana(self, **kwargs):
        if self.tree_helper_params[kwargs['which_tree']]['stemminess_indexes_Dana'] is None:
            stemminess_indexes = self.calc_stemminess_indexes_dana(self.trees[kwargs['which_tree']])
            self.tree_helper_params[kwargs['which_tree']]['stemminess_indexes_Dana'] = stemminess_indexes
            return stemminess_indexes
        else:
            return self.tree_helper_params[kwargs['which_tree']]['stemminess_indexes_Dana']

    def calc_stemminess_indexes_dana(self, tree):
        subtree_blsum_dict = {}
        nodes_height_dict = {}
        stem85_index_lst = []
        stem90_index_lst = []
        for node in tree.traverse(strategy="postorder"):
            if node.is_leaf():
                subtree_blsum_dict[node] = 0
                nodes_height_dict[node] = 0
            elif node.is_root():
                continue
            else:
                subtree_blsum_dict[node] = subtree_blsum_dict[node.children[0]] + subtree_blsum_dict[node.children[1]] + \
                                           node.children[0].dist + node.children[1].dist
                nodes_height_dict[node] = max(nodes_height_dict[node.children[0]] + node.children[0].dist,
                                              nodes_height_dict[node.children[1]] + node.children[1].dist)
                stem85_index_lst.append(node.dist / (subtree_blsum_dict[node] + node.dist))
                stem90_index_lst.append(node.dist / (nodes_height_dict[node]) + node.dist)

        if not stem85_index_lst or not stem90_index_lst:
            return 0,0

        return np.mean(stem85_index_lst), np.mean(stem90_index_lst)

    # add two feature funcs
    def calc_stemminess_indexes_dana_85(self, **kwargs):
        return self.features_stemminess_indexes_dana(**kwargs)[0]

    def calc_stemminess_indexes_dana_90(self, **kwargs):
        return self.features_stemminess_indexes_dana(**kwargs)[1]
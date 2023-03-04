import multiprocessing
import re
from io import StringIO
from subprocess import Popen, PIPE, STDOUT

import numpy as np
from Bio import AlignIO, Phylo
from Bio.Phylo.Consensus import *
from Bio.Phylo.TreeConstruction import *
from ete3 import *

import SharedConsts as SC
from SPR_generator import SPR_move


def generate_bootstrap_trees(data_set_number, algo, nbootrees):
    """
    :param data_set_number: data_set_number - also corresponds to which directory the data set is in within the data dir
    :param algo: could be 'nj' or 'upgma'
    :param nbootrees: the number of bootstrap trees to generate
    :return:trees: a biopython object storing all the bootstrap trees
    """
    msa_path = SC.PATH_TO_RAW_TREE_DATA / data_set_number / SC.MSA_FILE_NAME
    msa = AlignIO.read(msa_path, format="phylip")
    # bio python objects
    calculator = DistanceCalculator('identity')
    constructor = DistanceTreeConstructor(calculator, algo)

    # process and iterate the bootstrap trees to update hashtable
    trees_stream = StringIO()
    trees = bootstrap_trees(msa, nbootrees, constructor)
    Phylo.write(trees, trees_stream, "newick")
    lines_lst = trees_stream.getvalue().split('\n')[:-1]
    return lines_lst


def lst_to_str(leaves_lst):
    '''
    :param leaves_lst: a list of the species in a given msa
    :return: a string of the concatenated species names
    '''
    leaves_lst.sort()
    leaves_str = ''
    for i, e in enumerate(leaves_lst):
        leaves_str += e
    return leaves_str.encode()


def get_min_group(splitsHash, leaves_lst, sp_set, IsEven, max_k):
    if len(leaves_lst) > max_k:  # in case we weren't passed actual min group
        leaves_lst = list(sp_set - set(leaves_lst))
    leaves_str = lst_to_str(leaves_lst)

    # if len in even and we used the other half as a key, invert.
    if leaves_str not in splitsHash:
        if IsEven and len(leaves_lst) == max_k:
            leaves_lst = list(sp_set - set(leaves_lst))
            leaves_str = lst_to_str(leaves_lst)
    return leaves_str


def get_group_atts(sp_lst):
    max_k = len(sp_lst) // 2
    sp_set = set(sp_lst)
    IsEven = len(sp_lst) % 2 == 0
    return sp_set, IsEven, max_k


def update_splitsHash(original_splitsHash, btsrp_newicks_lst, species_lst):
    '''
    :param splitsHash: the dict with splits as keys and values as bootstrap value
    :param newicks_lst: lst of bootstrap trees as newick strings
    :param sp_lst: a list containing all the leaves names (strings) in the relevant msa
    :return:
    '''
    species_set, IsEven, max_k = get_group_atts(species_lst)
    ntrees = len(btsrp_newicks_lst)
    # iterate trees to update hashtable
    for newick in btsrp_newicks_lst:
        temp_splitsHash = {}
        tree = Tree(newick, format=1)
        # traverse each tree to index a n_appearance for each clade
        for node in tree.iter_descendants('postorder'):  # does not include root
            if node.is_leaf():
                leaf_str = lst_to_str([node.name])
                original_splitsHash[leaf_str] = 100
            else:
                leaves_lst = [leaf.name for leaf in node.get_leaves()]
                leaves_str = get_min_group(original_splitsHash, leaves_lst, species_set, IsEven, max_k)
                temp_splitsHash[leaves_str] = 1
        for split in temp_splitsHash:
            if original_splitsHash.get(split) is None:
                original_splitsHash[split] = 1
            else:
                original_splitsHash[split] += 1
    # update the split value as number of appearances/number of btsrp trees
    updated_splitsHash = {k: (round(v / ntrees, 4) if not v == 100 else v) for k, v in original_splitsHash.items()}
    return updated_splitsHash


def split_lookup(splitsHash, anyNode, data_set_number):
    '''
    :param splitsHash: the dict with splits as keys and values as bootstrap value
    :param anyNode: the relevant ete node for which we want the bstrap value
    :param data_set_number:
    :return: the bootstrap value for the split
    '''
    sp_lst = [sp.decode("utf-8")  for sp in splitsHash.keys() if str(sp).count('Sp') == 1]
    leaves_lst = [leaf.name for leaf in anyNode.get_leaves()]
    sp_set, IsEven, max_k = get_group_atts(sp_lst)
    if len(leaves_lst) == len(sp_set):  # namely if this is the NEW tree root
        new_node = anyNode.children[0]
        leaves_lst = [leaf.name for leaf in new_node.get_leaves()]
    leaves_str = get_min_group(splitsHash, leaves_lst, sp_set, IsEven, max_k)
    bstrap = splitsHash.get(leaves_str, 0)

    if len(leaves_lst) == 1:
        bstrap = 100

    return bstrap


"""
def generate_splitsHash(species_lst):
    '''
    split is defined as the ORDERED str containing the MINIMAL group of Species names.
    ties are broken to first in lexicographic order.
    :param species_lst: a list containing all the leaves names (strings) in the relevant msa
    :return: a dict with splits as keys and values as 0 to any internal branch, and 100 to any leaf
    '''
    splitsHash = {}
    for k in range(1, (len(species_lst)+1)//2):  # for the MINIMAL group we only need half of the species
        combs = combinations(species_lst, k)
        for comb_tup in combs:
            comb = lst_to_str(list(comb_tup))
            splitsHash[comb] = 0 if not len(comb_tup) == 1 else 100  # trivial splits are not relevant so get dummy value
    return splitsHash
"""


def get_likelihood_params(stat_path):
    params_dict = parse_phyml_stats_output(stat_path)
    freq, rates, pinv, alpha = [params_dict["fA"], params_dict["fC"], params_dict["fG"], params_dict["fT"]], [
        params_dict["subAC"], params_dict["subAG"], params_dict["subAT"], params_dict["subCG"], params_dict["subCT"],
        params_dict["subGT"]], params_dict["pInv"], params_dict["gamma"]

    return freq, rates, pinv, alpha


def parse_phyml_stats_output(stats_filepath):
    """
    :return: dictionary with the attributes - string typed. if parameter was not estimated, empty string
    """
    res_dict = dict.fromkeys(["ntaxa", "nchars", "ll",
                              "fA", "fC", "fG", "fT",
                              "subAC", "subAG", "subAT", "subCG", "subCT", "subGT",
                              "pInv", "gamma",
                              "path"], "")

    res_dict["path"] = stats_filepath
    try:
        with open(stats_filepath) as fpr:
            content = fpr.read()
            fpr.close()
        # likelihood
        res_dict["ll"] = re.search("Log-likelihood:\s+(.*)", content).group(1).strip()

        # gamma (alpha parameter) and proportion of invariant sites
        gamma_regex = re.search("Gamma shape parameter:\s+(.*)", content)
        pinv_regex = re.search("Proportion of invariant:\s+(.*)", content)
        if gamma_regex:
            res_dict['gamma'] = gamma_regex.group(1).strip()
        if pinv_regex:
            res_dict['pInv'] = pinv_regex.group(1).strip()

        # Nucleotides frequencies
        for nuc in "ACGT":
            nuc_freq = re.search("  - f\(" + nuc + "\)\= (.*)", content).group(1).strip()
            res_dict["f" + nuc] = nuc_freq

        # substitution frequencies
        for nuc1 in "ACGT":
            for nuc2 in "ACGT":
                if nuc1 < nuc2:
                    nuc_freq = re.search(nuc1 + " <-> " + nuc2 + "(.*)", content).group(1).strip()
                    res_dict["sub" + nuc1 + nuc2] = nuc_freq
    except:
        print("Error with:", res_dict["path"], res_dict["ntaxa"], res_dict["nchars"])
        return
    return res_dict


def parse_raxmlNG_content(content):
    """
    :param content:
    :return:dictionary with the attributes - string typed. if parameter was not estimated, empty string
    """
    try:
        res_dict = dict.fromkeys(["ll", "pInv", "gamma",
                                  "fA", "fC", "fG", "fT",
                                  "subAC", "subAG", "subAT", "subCG", "subCT", "subGT",
                                  "time"], "")

        # likelihood
        ll_re = re.search("Final LogLikelihood:\s+(.*)", content)
        if not ll_re and (
                re.search("BL opt converged to a worse likelihood score by",
                          content)):  # or re.search("failed", content)):
            tmp_likl = re.search("initial LogLikelihood:\s+(.*)", content)
            if tmp_likl is None:
                res_dict['ll'] = 0
            else:
                res_dict['ll'] = tmp_likl.group(1).strip()
        else:
            res_dict["ll"] = ll_re.group(1).strip()

            # gamma (alpha parameter) and proportion of invariant sites
            gamma_regex = re.search("alpha:\s+(\d+\.?\d*)\s+", content)
            pinv_regex = re.search("P-inv.*:\s+(\d+\.?\d*)", content)
            if gamma_regex:
                res_dict['gamma'] = gamma_regex.group(1).strip()
            if pinv_regex:
                res_dict['pInv'] = pinv_regex.group(1).strip()

            # Nucleotides frequencies
            nucs_freq = re.search("Base frequencies.*?:\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)",
                                  content)
            for i, nuc in enumerate("ACGT"):
                res_dict["f" + nuc] = nucs_freq.group(i + 1).strip()

            # substitution frequencies
            subs_freq = re.search(
                "Substitution rates.*:\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)",
                content)
            for i, nuc_pair in enumerate(["AC", "AG", "AT", "CG", "CT", "GT"]):
                res_dict["sub" + nuc_pair] = subs_freq.group(i + 1).strip()

            # Elapsed time of raxml-ng optimization
            rtime = re.search("Elapsed time:\s+(\d+\.?\d*)\s+seconds", content)
            if rtime:
                res_dict["time"] = rtime.group(1).strip()
            else:
                res_dict["time"] = 'no ll opt_no time'
    except Exception as e:
        tmp_likl = re.search("initial LogLikelihood:\s+(.*)", content)
        if tmp_likl is None:
            res_dict['ll'] = 0
        else:
            res_dict['ll'] = tmp_likl.group(1).strip()
        raise e

    return res_dict


def calc_likelihood(tree, msa_file, rates, pinv, alpha, freq):
    """
    :param tree: ETEtree OR a newick string
    :param msa_file:
    :param rates: as extracted from parse_raxmlNG_content() returned dict
    :param pinv: as extracted from parse_raxmlNG_content() returned dict
    :param alpha: as extracted from parse_raxmlNG_content() returned dict
    :param freq: as extracted from parse_raxmlNG_content() returned dict
    :param use_files: should generate optimized tree file
    :return: float. the score is the minus log-likelihood value of the tree
    """
    alpha = alpha if float(alpha) > 0.02 else 0.02
    model_line_params = 'GTR{rates}+I{pinv}+G{alpha}+F{freq}'.format(rates="{{{0}}}".format("/".join(rates)),
                                                                     pinv="{{{0}}}".format(pinv),
                                                                     alpha="{{{0}}}".format(alpha),
                                                                     freq="{{{0}}}".format("/".join(freq)))

    tree_rampath = "/dev/shm/" + msa_file.split("/")[-1] + "tree" + str(
        os.getpid())  # the var is the str: tmp{dir_suffix}
    no_files = '--nofiles'

    try:
        with open(tree_rampath, "w") as fpw:
            fpw.write(tree)
            fpw.close()
        p = Popen(
            [SC.RAXML_NG_SCRIPT, '--evaluate', '--msa', msa_file, '--threads', '1', '--opt-branches', 'on',
             '--opt-model',
             'off', '--model', model_line_params, no_files, '--tree', tree_rampath, '--redo', '--prefix', tree_rampath],
            stdout=PIPE, stdin=PIPE, stderr=STDOUT)

        raxml_stdout = p.communicate()[0]
        raxml_output = raxml_stdout.decode()
        res_dict = parse_raxmlNG_content(raxml_output)
        ll = res_dict['ll']

    # check for 'rare' alpha value error, run again with different alpha
    except AttributeError:
        # float 0.5-8
        new_alpha = 0.1
        model_line_params = 'GTR{rates}+I{pinv}+G{alpha}+F{freq}'.format(rates="{{{0}}}".format("/".join(rates)),
                                                                         pinv="{{{0}}}".format(pinv),
                                                                         alpha="{{{0}}}".format(new_alpha),
                                                                         freq="{{{0}}}".format("/".join(freq)))
        p = Popen(
            [SC.RAXML_NG_SCRIPT, '--evaluate', '--msa', msa_file, '--threads', '1', '--opt-branches', 'on',
             '--opt-model',
             'off', '--model', model_line_params, no_files, '--tree', tree_rampath, '--redo', '--prefix',tree_rampath],
            stdout=PIPE, stdin=PIPE, stderr=STDOUT)

        raxml_stdout = p.communicate()[0]
        raxml_output = raxml_stdout.decode()

        try:
            res_dict = parse_raxmlNG_content(raxml_output)
            ll = res_dict['ll']
        except:
            print("XXXXXXXXXXXXXXXXXX RaxML unkown error, probably due to invalid tree XXXXXXXXXXXXXXXX")
            print(raxml_output)
            ll = -12320.55555
    except Exception as e:
        print(msa_file)
        print(e)
        print("exception with likihood calc")
    finally:
        opt_tree = None

    return float(ll), opt_tree  # changed to return a num not a str


def msa_to_species_lst(data_set_number):
    '''
    :param data_set_number:
    :return: a list containing all the leaves names (strings) in the relevant msa
    '''
    msa_file = SC.PATH_TO_RAW_TREE_DATA / data_set_number / SC.MSA_FILE_NAME
    msa = AlignIO.read(msa_file, "phylip-relaxed")
    sp_lst = [seq.id for seq in msa]
    sp_lst.sort()
    return sp_lst


def get_newick_tree(tree_str):
    """
    :param tree_str: newick tree string or txt file containing one tree
    :return:tree: a string of the tree in ete3.Tree format
    """
    if os.path.exists(tree_str):
        with open(tree_str, 'r') as tree_fpr:
            tmp_tree = tree_fpr.read().strip()
            tree = Tree(newick=tmp_tree, format=1)
            tree_fpr.close()
    else:
        try:
            tree = Tree(newick=tree_str, format=1)
        except:
            raise Exception('Your input string is not a valid newick formatted tree or a path to one')
    return tree


def enforce_feature_data_type(results, non_float_features):
    """
    this function enforces data types for results in dict format. everything not in non_float_features will be a float.
    others will be strings.
    :param results: results in dict format
    :param non_float_features: features that are not in float form
    :return: the results dict enforced
    """
    keys = [name for name in results.keys()]
    for name in keys:
        temp_res = results[name]
        if name in non_float_features:
            if type(temp_res) != str:
                results[name] = str(results[name])
        else:
            if type(temp_res) != float:
                results[name] = float(results[name])
    return results


def pool_worker_wrapper(params):
    results = []
    for param in params:
        current_tree_obj = param['current_tree_obj']
        move = param['move']
        data_set_number = param['data_set_number']
        should_transform_results = param['should_transform_results']
        queue = param['queue']
        calculation_flag = param['calculation_flag']
        result_format = param['result_format']
        return_tree_string = param['return_tree_string']
        tool_box_instance = param['tool_box_instance']
        normalization_factor = param['normalization_factor']
        split_hash_dict = param['split_hash_dict']
        res = tool_box_instance.extract_features(current_tree_obj=current_tree_obj, move=move,
                                                 data_set_number=data_set_number, calculation_flag=calculation_flag,
                                                 result_format=result_format, return_tree_string=return_tree_string,
                                                 should_transform_results=should_transform_results, queue=queue,
                                                 normalization_factor=normalization_factor,
                                                 split_hash_dict=split_hash_dict)
        results.append(res)
    return results


def extract_all_neibours_features_multiprocessing(current_tree_obj, tool_box_instance, number_of_cpus,
                                                  data_set_num=None, should_transform_results=True, queue=None,
                                                  calculation_flag='all', result_format='dict',
                                                  return_tree_string=False, all_moves=None, normalization_factor=1,
                                                  split_hash_dict=None):
    """
    this function is an wrapper function for extract_ features. it is meant for calculating all neibours of a given tree
    using multiprocessing.
    also note that if move list was injected here than order of results is guaranteed.
    :return: the same as extract_features
    """
    param_list = []
    chunk_size = 10
    all_moves = SPR_move.get_moves_from_obj(current_tree_obj) if all_moves is None else all_moves
    for move in all_moves:
        param_dict = {'current_tree_obj': current_tree_obj, 'move': move, 'data_set_number': data_set_num,
                      'should_transform_results': should_transform_results, 'queue': queue,
                      'calculation_flag': calculation_flag, 'result_format': result_format,
                      'return_tree_string': return_tree_string, 'tool_box_instance': tool_box_instance,
                      'normalization_factor': normalization_factor, 'split_hash_dict':split_hash_dict}
        param_list.append(param_dict)
    param_list_partitioned = [list(i) for i in np.array_split(param_list, max(len(param_list)//chunk_size, 1))]
    with multiprocessing.Pool(max((number_of_cpus - 1), 1)) as pool:
        res = pool.map(pool_worker_wrapper, param_list_partitioned)
    results = [j for i in res for j in i]
    return results

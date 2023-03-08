import pathlib

from SharedConsts import SPR_RADIUS
from ete3 import *


class Edge:
    def __init__(self, node_a, node_b):
        self.node_a = node_a
        self.node_b = node_b

    def __str__(self):
        return ("[a={a} b={b}]".format(a=self.node_a, b=self.node_b))

    def __eq__(self, other):
        """Overrides the default implementation"""
        if ((self.node_a == other.node_a) and (self.node_b == other.node_b)) or (
                (self.node_b == other.node_a) and (self.node_a == other.node_b)):
            return True
        else:
            return False


def add_internal_names(original_tree):
    for i, node in enumerate(original_tree.traverse()):
        if not node.is_leaf():
            node.name = "N{}".format(i)
    return original_tree


def generate_tree_object(tree):
    if type(tree) in [pathlib.PosixPath, str]:
        starting_tree_object = Tree(newick=str(tree), format=1)
    else:
        starting_tree_object = tree
    add_internal_names(starting_tree_object)
    starting_tree_object.get_tree_root().name = "ROOT"
    return starting_tree_object


def get_list_of_edges(starting_tree):
    edges_list = []
    main_tree_root_pointer_cp = starting_tree.copy()
    for i, prune_node in enumerate(main_tree_root_pointer_cp.iter_descendants("levelorder")):
        if prune_node.up:
            edge = Edge(node_a=prune_node.name, node_b=prune_node.up.name)
            edges_list.append(edge)
    # for debugging purposes
    # print("total edges= {episode_num}".format(episode_num=episode_num + 1))
    return edges_list


def get_possible_spr_moves(edges_set, limit_radius_flag=False):
    possible_moves = []
    local_spr_radius = SPR_RADIUS if limit_radius_flag else sys.maxsize
    for i, edge1 in enumerate(edges_set):
        for j, edge2 in enumerate(edges_set):
            if abs(i - j) <= local_spr_radius:
                if not ((edge1.node_a == edge2.node_a) or (edge1.node_b == edge2.node_b) or (
                        edge1.node_b == edge2.node_a) or (edge1.node_a == edge2.node_b)):
                    possible_moves.append((edge1, edge2))
    return possible_moves


def get_moves_from_obj(tree_object):
    tree = generate_tree_object(tree_object)
    edges_list = get_list_of_edges(tree)
    moves = get_possible_spr_moves(edges_list, SPR_RADIUS is not None)

    return moves


def get_moves_from_file(tree_file):
    tree = generate_tree_object(str(tree_file))
    edges_list = get_list_of_edges(tree)
    moves = get_possible_spr_moves(edges_list, SPR_RADIUS is not None)

    return moves, tree


def add_subtree_to_basetree(subtree_root, basetree_root, pruned_edge, regraft_edge, length_regraft_edge,
                            length_pruned_edge):
    future_sister_tree_to_pruned_tree = (basetree_root & regraft_edge.node_a).detach()
    new_tree_adding_pruned_and_future_sister = Tree()
    new_tree_adding_pruned_and_future_sister.add_child(subtree_root.copy(),
                                                       dist=length_pruned_edge)
    new_tree_adding_pruned_and_future_sister.add_child(future_sister_tree_to_pruned_tree, dist=length_regraft_edge / 2)
    new_tree_adding_pruned_and_future_sister.name = pruned_edge.node_b
    (basetree_root & regraft_edge.node_b).add_child(new_tree_adding_pruned_and_future_sister,
                                                    dist=length_regraft_edge / 2)

    if pruned_edge.node_b == "ROOT":
        (basetree_root.get_tree_root()).name = "NEWROOT"
        if len(basetree_root.children) == 1:
            newrootname = (basetree_root.children[0]).name
            (basetree_root & newrootname).delete()
            basetree_root.get_tree_root().name = newrootname

    basetree_root.unroot()
    return basetree_root


def generate_base_neighbour(base_tree, possible_move):
    output_tree, _, _, _, _ = generate_neighbour(base_tree, possible_move, generate_remaining_subtrees=False)
    return output_tree


def generate_neighbour(base_tree, possible_move, generate_remaining_subtrees=True):
    base_tree = base_tree.copy()  # not working on original tree
    pruned_edge, regraft_edge = possible_move
    length_regraft_edge = (base_tree & regraft_edge.node_a).dist
    length_pruned_edge = (base_tree & pruned_edge.node_a).dist
    if base_tree.get_common_ancestor(regraft_edge.node_a, pruned_edge.node_a).name == pruned_edge.node_a:
        new_base_tree = (base_tree & pruned_edge.node_a).detach()
        new_subtree_to_be_regrafted = base_tree
        if not (
                       new_subtree_to_be_regrafted & pruned_edge.node_b).name == new_subtree_to_be_regrafted.get_tree_root().name:
            new_subtree_to_be_regrafted.set_outgroup(new_subtree_to_be_regrafted & pruned_edge.node_b)
        (new_subtree_to_be_regrafted & pruned_edge.node_b).delete(preserve_branch_length=True)
        pruned_subtree = new_subtree_to_be_regrafted.copy()
        remaining_subtree = new_base_tree.copy()
        output_tree = add_subtree_to_basetree(new_subtree_to_be_regrafted, new_base_tree, pruned_edge, regraft_edge,
                                              length_regraft_edge, length_pruned_edge)

    else:
        pruned_subtree = (base_tree & pruned_edge.node_a).detach()
        (base_tree & pruned_edge.node_b).delete(preserve_branch_length=True)
        remaining_subtree = base_tree.copy()
        output_tree = add_subtree_to_basetree(pruned_subtree, base_tree, pruned_edge, regraft_edge, length_regraft_edge,
                                              length_pruned_edge)

    # just to return the desired output for our pipeline
    remaining_subtree_subtree1, remaining_subtree_subtree2 = None, None
    if generate_remaining_subtrees:
        remaining_subtree_subtree2 = remaining_subtree.copy()
        remaining_subtree_subtree1 = (remaining_subtree_subtree2 & regraft_edge.node_a).detach()
        (remaining_subtree_subtree2 & regraft_edge.node_b).delete(preserve_branch_length=True)

    return output_tree, pruned_subtree, remaining_subtree, remaining_subtree_subtree1, remaining_subtree_subtree2



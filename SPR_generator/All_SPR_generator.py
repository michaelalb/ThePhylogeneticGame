from pathlib import Path

from ete3 import *

from SPR_generator.SPR_move import get_moves_from_file, generate_neighbour


def all_SPR(path, tree_file):

    moves, tree = get_moves_from_file(tree_file)

    # create new dir for results
    if not os.path.exists(path / 'all_SPR'):
        os.makedirs(path / 'all_SPR')

    # generate ALL possible moves and save to files
    for move in moves:
        new_tree = generate_neighbour(tree, move)[0]
        outfile = path / 'all_SPR' / (move_to_str(move) + ".txt")
        new_tree.write(format=1, outfile=outfile)


def move_to_str(move):
    res = '(' + move[0].node_a + ',' + move[0].node_b + ')'
    res += ' , ' + '(' + move[1].node_a + ',' + move[1].node_b + ')'
    return res


if __name__ == '__main__':
    import SharedConsts as SC
    # dataset_path =  Path('some_path')
    dataset_path = Path().resolve() / 'tree_folder'

    # "masked_species_real_msa.phy_phyml_tree_bionj.txt"
    test_tree_file = dataset_path / SC.NJ_STARTING_TREE_FILE_NAME
    all_SPR(dataset_path, test_tree_file)

import SharedConsts as SC
from FeatureExtractor.FeatureExtractorGeneralToolBox import calc_likelihood, get_likelihood_params
from SPR_generator.SPR_move import get_moves_from_obj, generate_base_neighbour


class CheatingGreedyAgent:
    """
    an agent to implement the hill climb heuristic
    using the actual tree, NOT the features.
    Notice this class breaks the environment encapsulation
    """

    @staticmethod
    def hill_climb(data_set, start_tree):

        msa_file_path = str(SC.PATH_TO_RAW_TREE_DATA / str(data_set) / SC.MSA_FILE_NAME)
        stat_path = str(SC.PATH_TO_RAW_TREE_DATA / str(data_set) / SC.PHYML_PARAM_FILE_NAME)
        freq, rates, pinv, alpha = get_likelihood_params(stat_path)

        current_tree = start_tree
        max_likelihood, _ = calc_likelihood(start_tree.write(format=1, format_root_node=True),
                                            msa_file_path, rates, pinv, alpha, freq)
        travel_log = [max_likelihood]
        step_taken = True
        for _ in range(SC.HORIZON):

            if not step_taken:
                # reached local maximum
                travel_log.append(max_likelihood)
                continue

            best_tree_so_far = current_tree
            step_taken = False
            possible_actions = get_moves_from_obj(current_tree)

            for action in possible_actions:
                tree = generate_base_neighbour(current_tree, action)
                likelihood, _ = calc_likelihood(tree.write(format=1, format_root_node=True),
                                                msa_file_path, rates, pinv, alpha, freq)

                if likelihood > max_likelihood:
                    max_likelihood = likelihood
                    best_tree_so_far = tree
                    step_taken = True

            travel_log.append(max_likelihood)
            current_tree = best_tree_so_far

        # check travel log is increasing as we expect
        assert len(travel_log) == SC.HORIZON + 1
        assert all(travel_log[i] <= travel_log[i + 1] for i in range(len(travel_log) - 1))

        return travel_log

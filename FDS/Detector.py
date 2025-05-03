from __future__ import division

import logging
from FDS import constants
import FDS.FDS as FDS

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(constants.log_level)


class Detector:
    """
    In this class we will implement all the methods related to detectors
    """

    def __init__(self):
        self.check_detector_points = 0

    def check_if_landscape_has_changed(self, fds, dataset):
        population = fds.gather_hyperspheres(level=-1)
        population = [sample for sample in population if sample.is_hypersphere_exploited]
        population += fds.trackers

        if not population:
            print("NO POPULATION TO CHECK THE DETECTORS")
            import ipdb; ipdb.set_trace()

        is_landscape_changed = False
        # TODO: Select the subpopulation to evaluate the change in the landscape
        for sample in population[0:min(len(population), constants.max_detector_evaluation_values)]:
            self.check_detector_points += 1
            fitness_solution, is_landscape_change, are_max_evaluations_reached = dataset.evaluate_fitness(sample.fitness_coordinates, sample.id, "Detector")
            # Detect when peaks move
            if sample.fitness != fitness_solution:
                # INITIALIZE FDS
                if constants.reset_method in ["fds_from_scratch"]:
                    dataset.searched_solutions = {}
                    fds = FDS.FDS(dataset)
                elif constants.reset_method in ["only_ils", "only_trackers_and_parents"]:
                    # LOGGER.info("self.counter: " + str(dataset.counter))
                    fds.re_init_fds()
                    LOGGER.info("FDS HAS BEEN RE-INITIALIZED")
                    fds.re_evaluate_solutions()
                    LOGGER.info("FDS HAS BEEN RE-EVALUATED")

                    constants.k_levels_min = constants.init_k_levels_min

                    # Fully decompose hyperspheres
                    while fds.current_hypersphere is not None:
                        fds.decomposition_hypersphere()
                        fds.decompose_FDS()
                break

        return fds, is_landscape_changed


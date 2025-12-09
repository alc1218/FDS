from __future__ import division

import time
import copy as copy
import logging
import numpy as np
from FDS import constants
from FDS import utils
from FDS.exploitation import Intensification
from FDS.exploration import Decomposition, Hypersphere as H

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(constants.log_level)


class FDS(Intensification.Intensification, Decomposition.Decomposition):

    # Define dataset to be optimized
    dataset = None

    # Define ILS stop criterion
    stop_criterion = constants.ils_max_iterations_stop_criterion

    initial_hypersphere = None
    current_hypersphere = None

    best_solution_coordinates_HSD = []
    best_solution_fitness_HSD = float('Inf')

    best_solution_coordinates_ILS = []
    best_solution_fitness_ILS = float('Inf')

    best_solution_coordinates = float('Inf')
    best_solution_fitness = float('Inf')

    is_fds_finished = False

    historic_of_searched_solutions_in_previous_landscapes = []
    searched_solutions_in_previous_landscapes = []

    nb_trackers = 0
    historic_of_trackers_that_did_not_locate_back_the_optima = [[]]

    # hyperspheres_population = []
    trackers = []
    historic_of_severities = []

    period_status = 0

    rotation_matrix = None
    # rotation_matrix = numpy.linalg.inv(rotation_matrix)

    def __init__(self, dataset):
        """

        @param dataset: dataset to be optimized
        """

        self.lapse_time = []

        self.ILS_gradient_stepsize = []

        self.ILS_num_evaluations = []

        # Integrate the parameters
        self.dataset = dataset

        self.define_initial_hypersphere()

        # Fully decompose hyperspheres
        if constants.decompose_tree in ["full"]:
            st = time.time()
            while self.current_hypersphere is not None:
                is_landscape_changed, are_max_evaluations_reached = self.decomposition_hypersphere()
                self.decompose_FDS()
            end = time.time()
            LOGGER.info(f"Decompose {constants.k_levels_max} level/s took: {end - st} seconds")

            self.current_hypersphere = self.initial_hypersphere
            is_landscape_changed, are_max_evaluations_reached, current_hypersphere = self.main_intensification()

        self.dataset.hyperspheres_centers = self.gather_hyperspheres(level=-1)

        # Define population
        # self.population = []

        # Definition of select_last_x_solutions parameter in order to remove the last local minimum detected because of the change of landscape
        counter_iters = 0
        step_size_tmp = constants.step_size
        while step_size_tmp > constants.GAMMA_MIN:
            counter_iters += 1
            step_size_tmp *= constants.GAMMA_DECREASE_STEP

        # +1 on the ILS_center_point
        self.select_last_x_solutions = self.dataset.number_of_dimensions * 2 * counter_iters + constants.max_detector_evaluation_values + 1

        # import ipdb; ipdb.set_trace()

    def re_init_fds(self):


        self.dataset.times_scenario_has_changed += 1

        # Keep only if tracker has found a local minimum
        total_number_of_trackers = len(self.trackers)
        self.trackers = [hypersphere_instance for hypersphere_instance in self.trackers if hypersphere_instance.is_local_minimum]

        if constants.memory_based:
            hyperspheres = self.gather_hyperspheres(level=-1)
            hyperspheres = [hypersphere for hypersphere in hyperspheres if hypersphere.is_hypersphere_exploited]

            # self.searched_solutions_in_previous_landscapes += hyperspheres
            self.historic_of_searched_solutions_in_previous_landscapes.append(hyperspheres)
            self.searched_solutions_in_previous_landscapes = np.concatenate(self.historic_of_searched_solutions_in_previous_landscapes[-constants.periods_to_archives:]).tolist()

        self.dataset.searched_solutions = {}
        self.is_fds_finished = False

        self.best_solution_coordinates_ILS = []
        self.best_solution_fitness_ILS = float('Inf')

        self.best_solution_coordinates_HSD = []
        self.best_solution_fitness_HSD = float('Inf')

        self.best_solution_coordinates = []
        self.best_solution_fitness = float('Inf')

        # Reset FDS
        self.define_initial_hypersphere()

    def re_evaluate_solutions(self):

        create_empty_space_in_terminal()

        if constants.dataset_type in ["MPB"] and constants.is_show_peaks_information_active:
            LOGGER.info(f"globalMaximum: {sorted(self.dataset.benchmark.maximums(), key=lambda x: x[0], reverse=True)[0]}")
            LOGGER.info("")

        if constants.number_of_archives_cyclic_optima_to_be_searched > 0:
            self.trackers += np.concatenate(self.historic_of_trackers_that_did_not_locate_back_the_optima[-constants.number_of_archives_cyclic_optima_to_be_searched:]).tolist()

        nb_hyperspheres_initial = len(self.trackers)
        nb_hyperspheres = len(self.trackers)
        LOGGER.debug("nb_hyperspheres: " + str(nb_hyperspheres))

        if constants.dataset_type in ["MPB"] and constants.is_show_peaks_information_active:
            local_optima_found = []
            maximums = self.dataset.maximums
            for hypersphere_instance in self.trackers:
                unormalize_solution = self.dataset.unormalize_solution(hypersphere_instance.fitness_coordinates)
                for maximum in maximums:
                    if utils.calculate_distance(unormalize_solution, maximum[1]) < 1:
                        local_optima_found.append(maximum[1])
                        LOGGER.info("utils.calculate_distance(unormalize_solution, maximum[1]): " + str(
                            utils.calculate_distance(unormalize_solution, maximum[1]) / 100) + " hypersphere_instance.fitness " + str(
                            hypersphere_instance.fitness) + " maximum[0] " + str(
                            maximum[0]) + " unormalize_solution: " + str(
                            unormalize_solution) + " maximum[1]: " + str(maximum[1]))

            LOGGER.info("Unique len(local_optima_found): " + str(len(np.unique(local_optima_found, axis=0))))
            LOGGER.info("local_optima_found: " + str(np.unique(local_optima_found, axis=0)))

            LOGGER.debug("nb_hyperspheres:", nb_hyperspheres)

        # Evaluate local minimum points sort based on best performance in the previous period
        self.trackers.sort(key=lambda x: x.fitness)
        for hypersphere_instance in self.trackers:
            if constants.tracking_step_size > 0:
                hypersphere_instance.radius = constants.tracking_step_size
                hypersphere_instance.historic_solutions = []
                hypersphere_instance.historic_fitness = []
                hypersphere_instance.is_local_minimum = False
                hypersphere_instance.hypersphere_parent = None

            hypersphere_instance.fitness, is_landscape_changed, are_max_evaluations_reached = self.dataset.evaluate_fitness(hypersphere_instance.fitness_coordinates, hypersphere_instance.id, "H")
            hypersphere_instance.historic_solutions.append(hypersphere_instance.fitness_coordinates)
            hypersphere_instance.historic_fitness.append(hypersphere_instance.fitness)
            hypersphere_instance.historic_evaluation_step.append(self.dataset.number_of_evaluations - 1)

            self.save_best_solution(hypersphere_instance)

        self.trackers.sort(key=lambda x: x.fitness)

        if constants.dataset_type in ["MPB"]:
            previous_GAMMA_MIN = constants.GAMMA_MIN
            previous_fixed_gamma = constants.fixed_gamma
            previous_step_size = constants.step_size
            constants.GAMMA_MIN = constants.GAMMA_MIN_trackers_stagnation
            constants.fixed_gamma = constants.GAMMA_MIN * 10
            constants.step_size = constants.GAMMA_MIN * 10

            if constants.severity_prediction and self.historic_of_severities:
                severity_factor = np.median(self.historic_of_severities) * 100
                constants.step_size = constants.step_size * severity_factor

        trackers_that_did_not_locate_back_the_optima = []
        trackers_that_did_locate_back_the_optima = []
        for hypersphere_instance in self.trackers:

            self.current_hypersphere = hypersphere_instance
            self.current_hypersphere.center = self.current_hypersphere.fitness_coordinates
            is_landscape_changed, are_max_evaluations_reached, current_hypersphere = self.main_intensification()

            if not self.current_hypersphere.is_local_minimum:
                trackers_that_did_not_locate_back_the_optima.append(self.current_hypersphere)

        for tracker in trackers_that_did_not_locate_back_the_optima:
            self.trackers.remove(tracker)

        if constants.dataset_type in ["MPB"]:
            constants.GAMMA_MIN = previous_GAMMA_MIN
            constants.fixed_gamma = previous_fixed_gamma
            constants.step_size = previous_step_size

        # Append trackers in the historic of trackers that have not find back its optima
        trackers_that_did_not_locate_back_the_optima_not_repeated = []
        if constants.number_of_archives_cyclic_optima_to_be_searched > 0:
            for current_tracker in trackers_that_did_not_locate_back_the_optima:
                is_tracker_in_list = False
                for historic_trackers in self.historic_of_trackers_that_did_not_locate_back_the_optima:
                    for historic_tracker in historic_trackers:
                        if historic_tracker == current_tracker:
                            is_tracker_in_list = True
                            break
                    if is_tracker_in_list:
                        break
                if not is_tracker_in_list:
                    trackers_that_did_not_locate_back_the_optima_not_repeated.append(current_tracker)

            self.historic_of_trackers_that_did_not_locate_back_the_optima.append(trackers_that_did_not_locate_back_the_optima_not_repeated)
            self.historic_of_trackers_that_did_not_locate_back_the_optima = self.historic_of_trackers_that_did_not_locate_back_the_optima[-constants.number_of_archives_cyclic_optima_to_be_searched:]

        # Estimate landscape severity
        if constants.severity_prediction:
            for tracker in self.trackers:
                if tracker.is_local_minimum:
                    array_1 = np.asarray(tracker.historic_solutions[0])
                    array_2 = np.asarray(tracker.historic_solutions[-1])
                    severity = utils.calculate_distance(array_1, array_2)
                    self.historic_of_severities.append(severity)

        if constants.decompose_tree in ["full"]:
            # Search root hypersphere
            self.current_hypersphere = self.initial_hypersphere
            if not self.searched_solutions_in_previous_landscapes:
                is_landscape_changed, are_max_evaluations_reached, current_hypersphere = self.main_intensification()
        elif constants.decompose_tree in ["stepwise"]:
            # Search root hypersphere
            self.current_hypersphere = self.initial_hypersphere

        nb_hyperspheres = len(self.trackers)
        LOGGER.debug("nb_hyperspheres: " + str(nb_hyperspheres))

        if constants.dataset_type in ["MPB"] and constants.is_show_peaks_information_active:
            local_optima_found = []
            maximums = self.dataset.benchmark.maximums()
            for hypersphere_instance in self.trackers:
                unormalize_solution = self.dataset.unormalize_solution(hypersphere_instance.fitness_coordinates)
                for maximum in maximums:
                    if utils.calculate_distance(unormalize_solution, maximum[1]) < 10:
                        local_optima_found.append(maximum[1])
                        LOGGER.info("utils.calculate_distance(unormalize_solution, maximum[1]): " + str(
                            utils.calculate_distance(unormalize_solution, maximum[1]) / 100) + " hypersphere_instance.fitness " + str(
                            hypersphere_instance.fitness) + " maximum[0] " + str(
                            maximum[0]) + " unormalize_solution: " + str(
                            unormalize_solution) + " maximum[1]: " + str(maximum[1]))

            LOGGER.info("Unique len(local_optima_found): " + str(len(np.unique(local_optima_found, axis=0))))
            LOGGER.info("local_optima_found: " + str(np.unique(local_optima_found, axis=0)))

            if self.nb_trackers != len(local_optima_found) and len(local_optima_found) > 0:
                self.nb_trackers = len(local_optima_found)

        create_empty_space_in_terminal()
        self.dataset.data_to_be_stored["num_trackers_after_change"].append(len(self.trackers))

    def save_best_solution(self, hypersphere_instance):
        if hypersphere_instance.fitness is not None:
            if hypersphere_instance.fitness < self.best_solution_fitness_ILS:
                self.best_solution_fitness_ILS = hypersphere_instance.fitness
                self.best_solution_coordinates_ILS = hypersphere_instance.fitness_coordinates
            if hypersphere_instance.fitness < self.best_solution_fitness_HSD:
                self.best_solution_fitness_HSD = hypersphere_instance.fitness
                self.best_solution_coordinates_HSD = hypersphere_instance.fitness_coordinates
            if not hypersphere_instance.hypersphere_children:
                if hypersphere_instance.fitness < self.best_solution_fitness:
                    self.best_solution_fitness = hypersphere_instance.fitness
                    self.best_solution_coordinates = hypersphere_instance.fitness_coordinates
                    # self.current_hypersphere = hypersphere_instance

    def define_initial_hypersphere(self):
        
        # Define rotation matrix to rotate hypespheres decomposition
        A = np.random.random((self.dataset.number_of_dimensions,self.dataset.number_of_dimensions))
        Q, _ = np.linalg.qr(A)
        self.rotation_matrix = Q
        
        # Define Radius
        tmp_radius = (self.dataset.upper_bound_norm - self.dataset.lower_bound_norm)
        tmp_radius = tmp_radius / 2

        # Define temporal Center for the first Current HyperSphere
        tmp_center = []

        # Init the center
        for i in range(0, self.dataset.number_of_dimensions):
            center_of_sphere_coordinate = self.dataset.lower_bound_norm + (
                        (self.dataset.upper_bound_norm - self.dataset.lower_bound_norm) / 2)
            tmp_center.append(center_of_sphere_coordinate)
            self.best_solution_coordinates_ILS.append(center_of_sphere_coordinate)
            self.best_solution_coordinates_HSD.append(center_of_sphere_coordinate)

        tmp_center = np.asarray(tmp_center)
        self.best_solution_coordinates_ILS = np.asarray(self.best_solution_coordinates_ILS)
        self.best_solution_coordinates_HSD = np.asarray(self.best_solution_coordinates_HSD)

        current_hypersphere_level = constants.hypersphere_root_level
        self.initial_hypersphere = H.Hypersphere(tmp_center, tmp_radius, current_hypersphere_level, None)
        self.current_hypersphere = self.initial_hypersphere

        self.current_hypersphere.fitness_coordinates = copy.deepcopy(self.current_hypersphere.center)
        self.current_hypersphere.fitness, is_landscape_changed, are_max_evaluations_reached = self.dataset.evaluate_fitness(self.current_hypersphere.center, "1_0", "H")
        self.current_hypersphere.historic_solutions.append(self.current_hypersphere.center)
        self.current_hypersphere.historic_fitness.append(self.current_hypersphere.fitness)
        self.current_hypersphere.historic_evaluation_step.append(self.dataset.number_of_evaluations - 1)

        self.save_best_solution(self.current_hypersphere)

        self.nb_of_spheres_visited = 0
        self.int_nb_of_times_enter_max_level = 0


def create_empty_space_in_terminal():
    for sample in range(5):
        LOGGER.debug("")

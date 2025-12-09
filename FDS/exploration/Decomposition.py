from FDS import constants
from FDS.exploration import utils_decomposition, Hypersphere as H
import numpy as np
from numba import jit
import logging
from scipy.spatial import distance as distance_scipy
import time
import copy
import math
from FDS import utils

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(constants.log_level)


@jit(nopython=True)  # Set "nopython" mode for best performance, equivalent to @njit
def compute_distance_in_numba(array_1, array_2):
    return np.sqrt((np.square(array_1 - array_2).sum(axis=2)))


class Decomposition(utils_decomposition.UtilsDecomposition):
    # ##################################################################################################################
    # ################################################### EXPLORATION ##################################################
    # ############################################ HYPERSPHERE DECOMPOSITION ###########################################
    # ##################################################################################################################
    def decomposition_hypersphere(self):
        """

        @return:
        """

        # Creation of the children hyperspheres
        self.create_children_hyperspheres()

        # Inflation & evaluation of the hyperspheres
        is_landscape_changed, are_max_evaluations_reached = self.inflate_hyper_sphere_and_compute_fitness()

        # Select next child to be processed best of selected strategy
        # self.select_child_hypersphere()

        return is_landscape_changed, are_max_evaluations_reached

    def create_children_hyperspheres(self):
        if constants.decomposition_type in ["FDS"]:
            shape = constants.datasets[constants.dataset_type]["dimensions"] * 2
            sphere_created_dimension = 1

            for i in range(1, shape + 1):
                sub_hyper_sphere = H.Hypersphere(self.current_hypersphere.center,
                                                 self.current_hypersphere.radius / constants.RADIUS_RATE,
                                                 self.current_hypersphere.current_hypersphere_level + 1,
                                                 self.current_hypersphere)
                self.current_hypersphere.hypersphere_children.append(sub_hyper_sphere)

                if i % 2 == 0:
                    # Create the "PLUS" hypersphere
                    tmp_dim = (self.current_hypersphere.center[
                        sphere_created_dimension - 1]) + self.current_hypersphere.radius - (
                                          self.current_hypersphere.radius / constants.RADIUS_RATE)
                    self.current_hypersphere.hypersphere_children[i - 1].center[sphere_created_dimension - 1] = tmp_dim
                    sphere_created_dimension += 1
                    # LOGGER.info("Create the PLUS hypersphere: " + str(tmp_dim))
                else:
                    # Create the "Minus" hypersphere
                    tmp_dim = (self.current_hypersphere.center[sphere_created_dimension - 1]) - (
                                self.current_hypersphere.radius - (
                                    self.current_hypersphere.radius / constants.RADIUS_RATE))
                    self.current_hypersphere.hypersphere_children[i - 1].center[sphere_created_dimension - 1] = tmp_dim
                    # LOGGER.info("Create the MINUS hypersphere: " + str(tmp_dim))

                # Rotate already created hypersphere center
                if constants.apply_random_rotation:
                    point = self.current_hypersphere.hypersphere_children[i - 1].center
                    rotation_matrix = self.rotation_matrix
                    reference_point = self.current_hypersphere.center
                    self.current_hypersphere.hypersphere_children[i - 1].center = utils.rotate_point(point, rotation_matrix, reference_point)

        elif constants.decomposition_type in ["Maurey", "FDS_GMM"]:
            for metacenter in constants.metacenters:
                sub_hyper_sphere = H.Hypersphere(self.current_hypersphere.center,
                                                 self.current_hypersphere.radius / constants.RADIUS_RATE,
                                                 self.current_hypersphere.current_hypersphere_level + 1,
                                                 self.current_hypersphere)
                self.current_hypersphere.hypersphere_children.append(sub_hyper_sphere)

                products = [a * self.current_hypersphere.radius for a in metacenter]
                tmp_dim = np.asarray(self.current_hypersphere.center) + products
                self.current_hypersphere.hypersphere_children[- 1].center = tmp_dim.tolist()

    def inflate_hyper_sphere_and_compute_fitness(self):

        is_landscape_changed = False
        are_max_evaluations_reached= False

        # LOGGER.info("START EVALUATING HYPERSPHERE")
        for loop_list_sub_sphere in range(len(self.current_hypersphere.hypersphere_children)):
            self.current_hypersphere.hypersphere_children[loop_list_sub_sphere].inflate_hypersphere()
            if constants.decompose_tree in ["stepwise"]:
                is_landscape_changed, are_max_evaluations_reached = self.compute_evaluation_of_hypersphere(loop_list_sub_sphere)
                if is_landscape_changed or are_max_evaluations_reached:
                    return is_landscape_changed, are_max_evaluations_reached
        
        return is_landscape_changed, are_max_evaluations_reached

    def compute_evaluation_of_hypersphere(self, loop_list_sub_sphere):

        is_landscape_changed = False
        are_max_evaluations_reached= False

        def compute_ratio(coordinates_solution, solution_s, context):
            ratio_s = -np.float("Inf")
            if solution_s != float("Inf"):

                if constants.gradient_computation_based_on in ["best_solution_found_so_far"]:
                    ratio_s = utils.compute_gradient(context.best_solution_fitness_ILS, solution_s,
                                                     context.best_solution_coordinates_ILS, coordinates_solution)

                elif constants.gradient_computation_based_on in ["greatest_gradient_from_HSD"]:
                    ratio_s = utils.compute_gradient(context.best_solution_fitness_HSD, solution_s,
                                                     context.best_solution_coordinates_HSD, coordinates_solution)

                elif constants.gradient_computation_based_on in ["parent_solution"]:
                    ratio_s = utils.compute_gradient(context.current_hypersphere.fitness, solution_s,
                                                     context.current_hypersphere.fitness_coordinates,
                                                     coordinates_solution)

                elif constants.gradient_computation_based_on in ["closest_points"]:
                    closest_hypersphere_instances = context.get_closest_points(coordinates_solution,
                                                                               number_of_points=constants.number_of_closest_points)

                    ratio_s = []
                    for closest_hypersphere_instance in closest_hypersphere_instances:
                        ratio_s.append(utils.compute_gradient(closest_hypersphere_instance.fitness, solution_s,
                                                              closest_hypersphere_instance.fitness_coordinates,
                                                              coordinates_solution))

                    if constants.HSD_extra_evaluation_policy in ["maximum"]:
                        ratio_s = np.max(ratio_s)
                    elif constants.HSD_extra_evaluation_policy in ["average"]:
                        ratio_s = np.mean(ratio_s)
                    elif constants.HSD_extra_evaluation_policy in ["minimum"]:
                        ratio_s = np.min(ratio_s)

            return ratio_s

        s_coordinates_list = []
        solution_fitness_list = []
        if constants.HSD_evaluation_policy in ["only_center"]:
            s_coordinates_list.append(
                copy.deepcopy(self.current_hypersphere.hypersphere_children[loop_list_sub_sphere].center))

            fitness, is_landscape_changed, are_max_evaluations_reached = self.dataset.evaluate_fitness(s_coordinates_list[-1],
                                                                          self.current_hypersphere.hypersphere_children[
                                                                              loop_list_sub_sphere].id, "H")

            if is_landscape_changed or are_max_evaluations_reached:
                return is_landscape_changed, are_max_evaluations_reached

            solution_fitness_list.append(fitness)
            self.current_hypersphere.hypersphere_children[loop_list_sub_sphere].historic_solutions.append(
                s_coordinates_list[-1])
            self.current_hypersphere.hypersphere_children[loop_list_sub_sphere].historic_fitness.append(
                solution_fitness_list[-1])

        elif constants.HSD_evaluation_policy in ["3_points"]:
            # Evaluate center point
            s_coordinates_list.append(
                copy.deepcopy(self.current_hypersphere.hypersphere_children[loop_list_sub_sphere].center))

            fitness, is_landscape_changed, are_max_evaluations_reached = self.dataset.evaluate_fitness(s_coordinates_list[-1],
                                                                          self.current_hypersphere.hypersphere_children[
                                                                              loop_list_sub_sphere].id, "H")
            
            if is_landscape_changed or are_max_evaluations_reached:
                return is_landscape_changed, are_max_evaluations_reached
            
            solution_fitness_list.append(fitness)
            self.current_hypersphere.hypersphere_children[loop_list_sub_sphere].historic_solutions.append(
                s_coordinates_list[-1])
            self.current_hypersphere.hypersphere_children[loop_list_sub_sphere].historic_fitness.append(
                solution_fitness_list[-1])

            # Positive in all dimension (+ step size)
            s_coordinates_list.append(
                copy.deepcopy(self.current_hypersphere.hypersphere_children[loop_list_sub_sphere].center))

            # Negative in all dimension (- step size)
            s_coordinates_list.append(
                copy.deepcopy(self.current_hypersphere.hypersphere_children[loop_list_sub_sphere].center))

            for i in range(0, self.dataset.number_of_dimensions):
                s_coordinates_list[-2][i] = self.current_hypersphere.hypersphere_children[loop_list_sub_sphere].center[
                                                i] + (self.current_hypersphere.hypersphere_children[
                                                          loop_list_sub_sphere].radius / math.sqrt(
                    self.dataset.number_of_dimensions))
                s_coordinates_list[-1][i] = self.current_hypersphere.hypersphere_children[loop_list_sub_sphere].center[
                                                i] - (self.current_hypersphere.hypersphere_children[
                                                          loop_list_sub_sphere].radius / math.sqrt(
                    self.dataset.number_of_dimensions))

            fitness, is_landscape_changed, are_max_evaluations_reached = self.dataset.evaluate_fitness(s_coordinates_list[-2],
                                                                          self.current_hypersphere.hypersphere_children[
                                                                              loop_list_sub_sphere].id, "H")
            
            if is_landscape_changed or are_max_evaluations_reached:
                return is_landscape_changed, are_max_evaluations_reached
            
            solution_fitness_list.append(fitness)
            self.current_hypersphere.hypersphere_children[loop_list_sub_sphere].historic_solutions.append(
                s_coordinates_list[-2])
            self.current_hypersphere.hypersphere_children[loop_list_sub_sphere].historic_fitness.append(
                solution_fitness_list[-1])

            fitness, is_landscape_changed, are_max_evaluations_reached = self.dataset.evaluate_fitness(s_coordinates_list[-1],
                                                                          self.current_hypersphere.hypersphere_children[
                                                                              loop_list_sub_sphere].id, "H")
            
            if is_landscape_changed or are_max_evaluations_reached:
                return is_landscape_changed, are_max_evaluations_reached

            solution_fitness_list.append(fitness)
            self.current_hypersphere.hypersphere_children[loop_list_sub_sphere].historic_solutions.append(
                s_coordinates_list[-1])
            self.current_hypersphere.hypersphere_children[loop_list_sub_sphere].historic_fitness.append(
                solution_fitness_list[-1])

        elif constants.HSD_evaluation_policy in ["every_dimension_2_points"]:
            # Evaluate center point
            s_coordinates_list.append(
                copy.deepcopy(self.current_hypersphere.hypersphere_children[loop_list_sub_sphere].center))

            fitness, is_landscape_changed, are_max_evaluations_reached = self.dataset.evaluate_fitness(s_coordinates_list[-1],
                                                                          self.current_hypersphere.hypersphere_children[
                                                                              loop_list_sub_sphere].id, "H")
            
            if is_landscape_changed or are_max_evaluations_reached:
                return is_landscape_changed, are_max_evaluations_reached
            
            solution_fitness_list.append(fitness)
            self.current_hypersphere.hypersphere_children[loop_list_sub_sphere].historic_solutions.append(
                s_coordinates_list[-1])
            self.current_hypersphere.hypersphere_children[loop_list_sub_sphere].historic_fitness.append(
                solution_fitness_list[-1])

            for i in range(0, self.dataset.number_of_dimensions):
                # Positive dimension (+ step size)
                s_coordinates_list.append(
                    copy.deepcopy(self.current_hypersphere.hypersphere_children[loop_list_sub_sphere].center))

                s_coordinates_list[-1][i] = self.current_hypersphere.hypersphere_children[loop_list_sub_sphere].center[
                                                i] + (self.current_hypersphere.hypersphere_children[
                                                          loop_list_sub_sphere].radius / math.sqrt(
                    self.dataset.number_of_dimensions))

                fitness, is_landscape_changed, are_max_evaluations_reached = self.dataset.evaluate_fitness(s_coordinates_list[-1],
                                                                              self.current_hypersphere.hypersphere_children[
                                                                                  loop_list_sub_sphere].id, "H")
                
                if is_landscape_changed or are_max_evaluations_reached:
                    return is_landscape_changed, are_max_evaluations_reached

                solution_fitness_list.append(fitness)
                self.current_hypersphere.hypersphere_children[loop_list_sub_sphere].historic_solutions.append(
                    s_coordinates_list[-1])
                self.current_hypersphere.hypersphere_children[loop_list_sub_sphere].historic_fitness.append(
                    solution_fitness_list[-1])

                # Negative dimension (- step size)
                s_coordinates_list.append(
                    copy.deepcopy(self.current_hypersphere.hypersphere_children[loop_list_sub_sphere].center))

                s_coordinates_list[-1][i] = self.current_hypersphere.hypersphere_children[loop_list_sub_sphere].center[
                                                i] - (self.current_hypersphere.hypersphere_children[
                                                          loop_list_sub_sphere].radius / math.sqrt(
                    self.dataset.number_of_dimensions))

                fitness, is_landscape_changed, are_max_evaluations_reached = self.dataset.evaluate_fitness(s_coordinates_list[-1],
                                                                              self.current_hypersphere.hypersphere_children[
                                                                                  loop_list_sub_sphere].id, "H")
                
                if is_landscape_changed or are_max_evaluations_reached:
                    return is_landscape_changed, are_max_evaluations_reached

                solution_fitness_list.append(fitness)
                self.current_hypersphere.hypersphere_children[loop_list_sub_sphere].historic_solutions.append(
                    s_coordinates_list[-1])
                self.current_hypersphere.hypersphere_children[loop_list_sub_sphere].historic_fitness.append(
                    solution_fitness_list[-1])

        else:
            raise Exception("No HSD evaluation policy selected")

        ratios_s = []

        for s_coordinates, solution_fitness in zip(s_coordinates_list, solution_fitness_list):
            ratios_s.append(compute_ratio(s_coordinates, solution_fitness, self))

        # fitness
        min_idx = np.argmin(solution_fitness_list)
        self.current_hypersphere.hypersphere_children[loop_list_sub_sphere].fitness = solution_fitness_list[min_idx]

        self.current_hypersphere.hypersphere_children[loop_list_sub_sphere].fitness_coordinates = copy.deepcopy(
            (s_coordinates_list)[min_idx])

        # ration
        if constants.HSD_extra_evaluation_policy in ["maximum"]:
            self.current_hypersphere.hypersphere_children[loop_list_sub_sphere].best_ratio = np.max(ratios_s)
        elif constants.HSD_extra_evaluation_policy in ["average"]:
            self.current_hypersphere.hypersphere_children[loop_list_sub_sphere].best_ratio = np.mean(ratios_s)
        elif constants.HSD_extra_evaluation_policy in ["minimum"]:
            self.current_hypersphere.hypersphere_children[loop_list_sub_sphere].best_ratio = np.min(ratios_s)
        else:
            raise Exception("No HSD EXTRA evaluation policy selected ")

        # Get best ration
        self.save_best_solution(self.current_hypersphere.hypersphere_children[loop_list_sub_sphere])

        if constants.gradient_computation_based_on in ["best_solution_found_so_far", "greatest_gradient_from_HSD",
                                                       "closest_points"]:
            self.update_hyperspheres_ratios()

        self.nb_of_spheres_visited += 1

        return is_landscape_changed, are_max_evaluations_reached

    def select_child_hypersphere(self):
        def get_key_fitness(hypersphere_instance):
            """

            @param hypersphere_instance: Sample in list
            @return:
            """
            fitness_score = float('Inf')
            if not hypersphere_instance.is_hypersphere_exploited:
                fitness_score = hypersphere_instance.fitness
            return fitness_score

        def get_key_ration(hypersphere_instance):
            """

            @param hypersphere_instance: Sample in list
            @return:
            """
            best_ratio = -1
            if not hypersphere_instance.is_hypersphere_exploited:
                best_ratio = hypersphere_instance.best_ratio
            return best_ratio

        if constants.strategy in ["lowest_error"]:
            # Test fitness value in order to obtain the best solution (to navigate)
            self.current_hypersphere.hypersphere_children.sort(key=get_key_fitness)

        elif constants.strategy in ["biggest_gradient"]:
            # Now sort by ratio (the bigger the better)
            self.current_hypersphere.hypersphere_children.sort(key=get_key_ration, reverse=True)

        self.current_hypersphere = self.current_hypersphere.hypersphere_children[0]

    # ##################################################################################################################
    # ##################################################### MOVE UP ####################################################
    # ####################################### MANAGE MOVING THROUGH HYPERSPHERES #######################################
    # ##################################################################################################################
    def check_if_all_FDS_structure_has_finished(self):
        # Iterate through all the hypersphere nodes and check if all the FDS structure has been explored
        stack = [self.initial_hypersphere]
        self.is_fds_finished = True
        while stack:
            hypersphere_instance = stack.pop()
            if hypersphere_instance.hypersphere_children:
                stack += hypersphere_instance.hypersphere_children
            else:
                if hypersphere_instance.fitness != float(
                        'Inf') and not hypersphere_instance.is_hypersphere_exploited:
                    self.is_fds_finished = False

    def decompose_FDS(self):
        # Explore K minimum levels
        # start_time = time.time()
        self.current_hypersphere = None
        stack = [self.initial_hypersphere]
        while stack:
            hypersphere_instance = stack.pop(0)
            if hypersphere_instance.current_hypersphere_level < constants.k_levels_min:
                stack += hypersphere_instance.hypersphere_children
                if not hypersphere_instance.hypersphere_children:
                    self.current_hypersphere = hypersphere_instance
                    return
        # self.lapse_time.append(time.time() - start_time)
        # print("lapse_time: " + str(np.mean(self.lapse_time)))
        # lapse_time: 5.820026136424443e-06

    def select_next_node_to_be_explored(self):

        selected_hyperspheres_to_be_intensified = []
        is_landscape_changed = False
        are_max_evaluations_reached = False

        if constants.backtracking == "depth_first":
            while not self.is_fds_finished:
                # Gather population of possible hyperspheres search spaces
                if self.current_hypersphere.hypersphere_parent is not None:
                    hypersphere_instances = [hypersphere_instance for hypersphere_instance in
                                             self.current_hypersphere.hypersphere_parent.hypersphere_children if
                                             not hypersphere_instance.is_hypersphere_exploited]

                    if hypersphere_instances:
                        # Sort population based on most promising search space (sort inplace)
                        if constants.strategy in ["lowest_error"]:
                            hypersphere_instances.sort(key=lambda x: x.fitness)
                        elif constants.strategy in ["biggest_gradient"]:
                            hypersphere_instances.sort(key=lambda x: x.best_ratio, reverse=True)

                        # Assign the current hypersphere as the most promising one
                        self.current_hypersphere = hypersphere_instances[0]
                        break

                    else:
                        self.current_hypersphere = self.current_hypersphere.hypersphere_parent
                        self.current_hypersphere.is_hypersphere_exploited = True
                else:
                    self.is_fds_finished = True

        elif constants.backtracking == "best_first":
            # Gather population of possible hyperspheres search spaces
            st = time.time()
            population = self.gather_hyperspheres(level=-1)
            population = [sample for sample in population if
                            not sample.is_hypersphere_exploited and (population[0].center > 0).all() and (
                                        population[0].center < 1).all()]

            # st = time.time()
            current_population = self.gather_hyperspheres(level=-1)
            end = time.time()
            LOGGER.debug(f"Gather hyperspheres time: {end - st}")
            coordinates_solutions = []
            # tmp_counter = 0

            # for hypersphere_explored_instance in self.trackers:
            #     coordinates = hypersphere_explored_instance.fitness_coordinates
            #     coordinates_solutions.append(coordinates)

            # for hypersphere_explored_instance in current_population + self.searched_solutions_in_previous_landscapes + self.trackers:
            for hypersphere_explored_instance in current_population + self.trackers:
                if hypersphere_explored_instance.is_hypersphere_exploited:
                    # coordinates = hypersphere_explored_instance.fitness_coordinates
                    # coordinates_solutions.append(coordinates)
                    coordinates = hypersphere_explored_instance.center
                    coordinates_solutions.append(coordinates)

                    """
                    # All the final points of each explored hyperspheres that contained a local minimum
                    if hypersphere_explored_instance.is_local_minimum:
                        coordinates = hypersphere_explored_instance.fitness_coordinates
                        coordinates_solutions.append(coordinates)
                        # tmp_counter += 1
                    """

                    # All the final points of each explored hyperspheres
                    coordinates = hypersphere_explored_instance.fitness_coordinates
                    coordinates_solutions.append(coordinates)
                    # tmp_counter += 1

            # coordinates = hypersphere_explored_instance.historic_solutions
            # for sample in coordinates:
            #     coordinates_solutions.append(sample)

            # if tmp_counter > 15:
            #     LOGGER.info(f"CRAZY - tmp_counter: {tmp_counter}")
            #     import ipdb; ipdb.set_trace()

            coordinates_population = []
            for idx, hypersphere_instance in enumerate(population):
                coordinates_population.append(hypersphere_instance.center)

            # Computing the distances (vectorized manner)
            # distances = (np.sqrt((np.square(np.asarray(coordinates_solutions)[:, np.newaxis, :] - np.asarray(coordinates_population)).sum(axis=2)))).tolist()[0]

            # distances = (np.sqrt((np.square(np.asarray(coordinates_solutions)[:, np.newaxis, :] - np.asarray(coordinates_population)).sum(axis=2)))).sum(axis=0).tolist()
            # distances_numpy = (np.sqrt((np.square(np.asarray(coordinates_solutions)[:, np.newaxis, :] - np.asarray(coordinates_population)).sum(axis=2))))

            # NUMBA
            # array_1 = np.asarray(coordinates_solutions)[:, np.newaxis, :]
            # array_2 = np.asarray(coordinates_population)
            # distances_numba = compute_distance_in_numba(array_1, array_2)

            # SCIPY
            if coordinates_solutions and coordinates_population:
                array_1 = np.asarray(coordinates_solutions)
                array_2 = np.asarray(coordinates_population)
                st = time.time()
                distances_scipy = distance_scipy.cdist(array_1, array_2)
                end = time.time()
                LOGGER.debug(f"Distances calculation hyperspheres time: {end - st}")

                """
                # DOT PRODUCT
                def closest_node(node, nodes):
                    deltas = nodes - node
                    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
                    return dist_2

                array_1 = np.asarray(coordinates_solutions)
                array_2 = np.asarray(coordinates_population)
                """

                distances = distances_scipy

                # if not np.allclose(distances_numpy, distances_numba) or not np.allclose(distances_numpy, distances) or not np.allclose(distances_numba, distances):
                #     import ipdb; ipdb.set_trace()

                # import ipdb; ipdb.set_trace()

                # end = time.time()
                # LOGGER.info(f"2) Distance computation took: {end - st} seconds")

                # import ipdb; ipdb.set_trace()

                distances = np.min(distances, axis=0).tolist()
                if distances:
                    population = [x for _, x in
                                    sorted(zip(distances, population), key=lambda pair: pair[0], reverse=True)]
                    distances.sort(reverse=True)

                    # Assess top X performers
                    best_performance_idx = 0
                    if constants.top_distant_hyperspheres > 1:
                        performances = []
                        for sample in population[:constants.top_distant_hyperspheres]:
                            # performances.append(self.dataset.evaluate_fitness(sample.center, sample.id, "H"))
                            fitness, is_landscape_changed, are_max_evaluations_reached = self.dataset.evaluate_fitness(sample.center, sample.id, "H")
                            
                            if is_landscape_changed or are_max_evaluations_reached:
                                return is_landscape_changed, are_max_evaluations_reached, selected_hyperspheres_to_be_intensified
                            
                            performances.append(fitness)

                        """
                        top_population = [sample.center for sample in population[:10]]
    
                        distances_tmp = []
                        for idx_top_sample, top_sample in enumerate(top_population):
                            distances_tmp.append(np.linalg.norm(np.asarray(top_sample) - np.asarray(top_population), axis=1))
    
                        distances_tmp = np.asarray(distances_tmp)
                        distances_tmp.sort()
    
                        """
                        best_performance_idx = np.argmin(performances)
                        # print(f"LOWEST INTRA DISTANCE: {np.min(distances_tmp[:, 1])}, Initially selected distance: {distances[0]}, FINAL selected distance: {distances[best_performance_idx]}")

                    self.current_hypersphere = population[best_performance_idx]

                # if self.period_status < self.dataset.number_of_evaluations // self.dataset.mpb.period:
                #     LOGGER.info(f"np.asarray(coordinates_solutions)[:, np.newaxis, :].shape): {np.asarray(coordinates_solutions)[:, np.newaxis, :].shape}, np.asarray(coordinates_population).shape: {np.asarray(coordinates_population).shape}")
                #     self.period_status = self.dataset.number_of_evaluations // self.dataset.mpb.period
                #     import ipdb; ipdb.set_trace()

            elif not coordinates_solutions and coordinates_population:
                self.current_hypersphere = population[0]
        
        selected_hyperspheres_to_be_intensified = population[: constants.intensification_in_parallel]
        
        return is_landscape_changed, are_max_evaluations_reached, selected_hyperspheres_to_be_intensified
        
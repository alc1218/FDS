import logging
from FDS import constants

from FDS import utils_vis
if constants.is_visualization:
    import cv2
import os
from tabulate import tabulate
import numpy as np

if constants.dataset_type in ["MPB"]:
    from deap import base
    from deap.benchmarks import movingpeaks
    import random

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(constants.log_level)

class Dataset:
    searched_solutions = {}

    hyperspheres_centers = None

    def __init__(self, random_seed, output_folder):
        self.counter = 0
        self.is_landscape_changed = False

        self.number_of_dimensions = None
        self.upper_bound = None
        self.lower_bound = None
        self.upper_bound_norm = 1.0
        self.lower_bound_norm = 0.0
        self.benchmark = None
        self.toolbox = None

        self.times_scenario_has_changed = 0

        self.maximums = None

        self.number_of_evaluations = 0
        self.random_seed = random_seed

        output_file = os.path.join(output_folder, "_".join([str(self.random_seed), constants.dataset_type + ".txt"]))
        self.f = open(output_file, "a")

        # Video output
        if constants.is_visualization:
            output_video_path = os.path.join(output_folder, "_".join(["FINAL", constants.dataset_type + ".avi"]))
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter(output_video_path, fourcc, 5.0, (constants.width_ouput_video, constants.height_ouput_video))

        if constants.dataset_type in ["MPB"]:
            self.load_mpb()
        else:
            raise Exception("Dataset not defined: " + str(constants.dataset_type))

        if constants.dataset_type in ["MPB"]:
            self.data_to_be_stored = {
                "coefficient": [],
                "fitness_score": [],
                "diversity": [],
                "currentError": [],
                "offlineError": [],
                "search_step": [],
                "hypersphere_IDs": [],
                "peaks": [],
                "peaks_attraction": [],
                "random_seed": self.random_seed,
                "NDIM": None,
                "period": constants.datasets[constants.dataset_type]["period"],
                "num_trackers_before_change":[],
                "num_trackers_after_change":[],
                "num_tracked_peaks":[],
                "num_vizible_peaks":[],
                "peaks_closseness_to_edge":[],
                "peaks_closseness_to_peaks":[],
            }

    def load_mpb(self):
        scenario = movingpeaks.SCENARIO_2

        scenario["move_severity"] = constants.datasets[constants.dataset_type]["move_severity"]
        scenario["period"] = constants.datasets[constants.dataset_type]["period"]
        scenario["npeaks"] = constants.datasets[constants.dataset_type]["npeaks"]
        scenario["lambda_"] = 0.

        self.number_of_dimensions = constants.datasets[constants.dataset_type]["dimensions"]
        bounds = [scenario["min_coord"], scenario["max_coord"]]

        random.seed(self.random_seed)
        self.benchmark = movingpeaks.MovingPeaks(dim=self.number_of_dimensions, random=random, **scenario)
        self.toolbox = base.Toolbox()
        self.toolbox.register("evaluate", self.benchmark)

        self.upper_bound = bounds[1]
        self.lower_bound = bounds[0]

    def evaluate_fitness(self, solution, solution_id, search_step=""):
        """

        @param solution:
        @param search_step: Type of evaluation: Exploration (HSD), Exploitation (ILS) or Detector (re-evaluation point)
        @return:
        """

        # LOGGER.debug("solution: " + str(solution))
        solution_out_of_bounds = False
        is_landscape_changed = False
        are_max_evaluations_reached = False

        # LOGGER.debug("solution: " + str(solution))
        for idx, i in enumerate(solution):
            if constants.penalization_based_on in ["inf"]:
                if i < self.lower_bound_norm or i > self.upper_bound_norm:
                    fitness_value = None
                    solution_out_of_bounds = True
                    break  # Skip extra dimensions iterations
            elif constants.penalization_based_on in ["boundary"]:
                if i < self.lower_bound_norm:
                    solution[idx] = self.lower_bound_norm
                elif i > self.upper_bound_norm:
                    solution[idx] = self.upper_bound_norm
            elif constants.penalization_based_on in ["mirror"]:
                if i < self.lower_bound_norm:
                    solution[idx] = self.lower_bound_norm - solution[idx]
                elif i > self.upper_bound_norm:
                    solution[idx] = solution[idx] - self.upper_bound_norm
            else:
                raise Exception("No penalization option chosen: " + str(constants.penalization_based_on))

        if not solution_out_of_bounds:
            if constants.dataset_type in ["MPB"]:

                fitness_value = None
                if constants.check_if_solution_has_been_searched and search_step not in ["Detector"]:
                    if self.searched_solutions.get(str(solution), None) is not None:
                        self.counter += 1
                    if str(solution) in self.searched_solutions:
                        fitness_value = self.searched_solutions.get(str(solution), None)["fitness_value"]
                        current_error = 0
                        offline_error = 0
                if fitness_value is None:
                    solution_unormalized = self.unormalize_solution(solution)

                    if constants.dataset_type in ["MPB"]:
                        # fitness_value, peak_attraction_idx = self.toolbox.evaluate(solution_unormalized)
                        fitness_value = self.toolbox.evaluate(solution_unormalized)[0]
                        peak_attraction_idx = None
                        current_error = self.benchmark.currentError()
                        offline_error = self.benchmark.offlineError()

                    self.number_of_evaluations += 1

                    if constants.record_pickle:
                        self.data_to_be_stored["coefficient"].append(solution)
                        self.data_to_be_stored["fitness_score"].append(fitness_value)
                        if constants.dataset_type in ["MPB"]:
                            # self.data_to_be_stored["diversity"].append(movingpeaks.diversity(self.data_to_be_stored["coefficient"]))
                            self.data_to_be_stored["currentError"].append(current_error)
                            self.data_to_be_stored["offlineError"].append(offline_error)
                            self.data_to_be_stored["search_step"].append(search_step)
                            self.data_to_be_stored["hypersphere_IDs"].append(solution_id)
                            self.data_to_be_stored["peaks_attraction"].append(peak_attraction_idx)

                    # Transform problem based on minimization or maximization operation
                    fitness_value = fitness_value * constants.datasets[constants.dataset_type]["make_problem_as_minimization"]

                    if constants.check_if_solution_has_been_searched:

                        self.searched_solutions[str(solution)] = {
                            "fitness_value": fitness_value,
                            "evaluation_number": self.number_of_evaluations,
                        }

                    # Write solution to output file
                    if constants.write_logs_to_file:
                        self.f.write(str(fitness_value) + "" + str(solution) + "\n")

                if constants.dataset_type in ["MPB"] and (self.number_of_evaluations + 1) % self.benchmark.period == 0:
                    self.maximums = self.benchmark.maximums()

                is_landscape_changed = self.check_if_landscape_has_changed()

                are_max_evaluations_reached = self.check_if_max_evaluations_are_reached()

                if constants.is_visualization:
                    if constants.dataset_type in ["MPB"]:
                        frame = utils_vis.create_frame(self.searched_solutions, self.benchmark.maximums(), self.hyperspheres_centers, self.toolbox)
                    else:
                        raise NameError(f"Not available visualization for this problem: {constants.dataset_type}")
                    self.out.write(frame)
            else:
                raise Exception("Dataset not defined: " + str(constants.dataset_type))

        if self.number_of_evaluations % constants.verbose == 0:
            if constants.dataset_type in ["MPB"]:
                LOGGER.info(str([[self.number_of_evaluations, solution_id, fitness_value, current_error, offline_error, solution.tolist()]]))

            if constants.dataset_type in ["MPB"] and self.number_of_evaluations % self.benchmark.period == 0:
                LOGGER.critical("LANDSCAPE CHANGED " + str(round(self.number_of_evaluations / self.benchmark.period)) + " times")

        return fitness_value, is_landscape_changed, are_max_evaluations_reached

    def evaluate_fitness_dummy(self, solution):
        solution_unormalized = self.unormalize_solution(solution)
        fitness_value, peak_attraction_idx = self.toolbox.evaluate(solution_unormalized, count=False)
        return fitness_value, peak_attraction_idx

    def unormalize_solution(self, solution):
        return solution * (self.upper_bound - self.lower_bound) + self.lower_bound

    def check_if_landscape_has_changed(self):
        # if self.number_of_evaluations > 1000 and self.number_of_evaluations % self.benchmark.period == 0:
        if constants.frequency_prediction and self.number_of_evaluations % self.benchmark.period == 0:
            self.is_landscape_changed = True
        return self.is_landscape_changed

    def check_if_max_evaluations_are_reached(self):
        are_max_evaluations_reached = False
        if self.number_of_evaluations >= constants.datasets[constants.dataset_type]["max_evaluations"]:
            are_max_evaluations_reached = True
        return are_max_evaluations_reached

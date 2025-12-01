import os
from datetime import datetime
import numpy as np
import sys

# Overall parameters
is_dynamic_activated = bool(os.environ.get('IS_DYNAMIC_ACTIVATED', 'true').lower() == "true")
# random_seeds = list(range(10))
random_seeds = [0]
# random_seeds = [31]
# random_seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
# random_seeds = list(range(100))  # 0.75
# random_seeds = list(range(10))

intensification_in_parallel = int(os.environ.get('INTENSIFICATION_IN_PARALLEL', 1))

log_level = "INFO"  # "NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"

apply_random_rotation = True

# HyperSphere Decomposition
GAMMA_MIN = float(os.environ.get('GAMMA_MIN', None))
GAMMA_MIN_trackers_stagnation = 1 * pow(10, -5)
GAMMA_DECREASE_STEP = float(os.environ.get('GAMMA_DECREASE_STEP', None))
EPSILON_DIVISION_0 = sys.float_info.epsilon

severity_prediction = bool(os.environ.get('SEVERITY_PREDICTION', 'true').lower() == "true")
frequency_prediction = bool(os.environ.get('FREQUENCY_PREDICTION', 'true').lower() == "true")
check_if_solution_has_been_searched = bool(os.environ.get('CHECK_IF_SOLUTION_HAS_BEEN_SEARCHED', 'false').lower() == "true")
is_visualization = False
write_logs_to_file = False
record_pickle = True
is_show_peaks_information_active = False
ils_visualization = False

max_process_batch = int(os.environ.get('MAX_PROCESS_BATCH', 20))
verbose = 5000

# Dataset
# "MPB"
dataset_type = str(os.environ.get('DATASET_TYPE', None))
datasets = {
    "MPB": {
        "dimensions": int(os.environ.get('DIMENSIONS', 5)),  # 5, 10, 20, 50, 100
        "move_severity": int(os.environ.get('MOVE_SEVERITY', 1)),  # 1, 2, 3, 4, 5, 6
        "period": int(os.environ.get('PERIOD', 5000)),  # 500, 1000, 2500, 5000, 10000
        "npeaks": int(os.environ.get('NPEAKS', 10)),  # 1, 5, 10, 20, 30, 40, 50, 100, 200
        "make_problem_as_minimization": -1,
        "max_evaluations": 5e4  # 5e3  # 5e4 # 5e5,
    }
}

# FDS parameters
reset_method = "only_trackers_and_parents"  # "fds_from_scratch", "only_ils", "only_trackers_and_parents"

top_distant_hyperspheres = 1
periods_to_archives = 1
number_of_archives_cyclic_optima_to_be_searched = 1

# Hypersphere min levels to be decomposed at the beginning of the search
init_k_levels_min = int(os.environ.get('HYPERSPHERE_LEVELS', None))
k_levels_min = int(os.environ.get('HYPERSPHERE_LEVELS', None))
# Hypersphere max levels to be decomposed during the search
k_levels_max = int(os.environ.get('HYPERSPHERE_LEVELS', None))
ils_max_iterations_stop_criterion = np.inf
max_detector_evaluation_values = 3
check_detectors_evaluations_frequency = 10

# step_size = GAMMA_MIN * 10
ILS_step_size = 0.0000000000001
step_size = float(os.environ.get('STEP_SIZE', None))
tracking_step_size = GAMMA_MIN_trackers_stagnation * 10

decompose_tree = "full"  # full, stepwise
child_policy = False
backtracking = "best_first"  # "depth_first", "best_first"

diversification_strategy = "space_diversification_distance_based"  # "None", "space_diversification_distance_based", "space_diversification_gradient_based"

memory_based = True
strategy = "biggest_gradient"  # "biggest_gradient",  "lowest_error"
gradient_computation_based_on = "parent_solution"  # "None", "best_solution_found_so_far", "parent_solution", "greatest_gradient_from_HSD", "closest_points"
number_of_closest_points = 1
ILS_method = "keep_best_solution"  # "select_best_among_steps", "keep_best_solution"

ILS_limitation = "local_minimum_and_fixed_gamma"  # "None", "local_minimum_and_gamma", "local_minimum_and_gamma_min", "local_minimum_and_fixed_gamma", "exploited_hyperspheres_and_step_size"
fixed_gamma = GAMMA_MIN * 10

modified_ILS = "direction_based_per_dimension_averaged"

penalization_based_on = "boundary"  # "inf", "boundary", "mirror"
decomposition_type = "FDS"  # "FDS", "Maurey", "Sampling", "FDS_GMM"
HSD_evaluation_policy = "only_center"  # "only_center", "3_points", "every_dimension_2_points"
HSD_extra_evaluation_policy = "maximum"  # "maximum", "average", "minimum"
cutout = 0

evaluations_for_tracking = -1

alpha = 1 * pow(10, 6)

if decomposition_type in ["FDS"]:
    RADIUS_RATE = 2.41421
    if datasets[dataset_type]["dimensions"] == 2:
        INFLATED_RATE = 1.75
        # INFLATED_RATE = 1.0
    elif datasets[dataset_type]["dimensions"] == 3:  # TODO: Set the appropriate inflation rate for 3D
            INFLATED_RATE = 1.75
    elif datasets[dataset_type]["dimensions"] == 4:
        INFLATED_RATE = 2.185  # 1.75
    elif datasets[dataset_type]["dimensions"] >= 5:
        INFLATED_RATE = 2.185  # 1.75
    else:
        raise Exception("CALCULATE THE PROPER INFLATION RATE")
    hypersphere_root_level = 1

# Output paths location
root_output_folder = "./outputs"
datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = os.path.join(root_output_folder, dataset_type, datetime_str)

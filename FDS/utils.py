from numba import jit
import numpy as np
import math
import os
import pickle
import logging

from FDS import constants
from FDS import utils_vis

logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(constants.log_level)


# ##################################################################################################################
# ################################################## Rotate point ##################################################
# ##################################################################################################################
def rotate_point(point, rotation_matrix, reference_point):

    translated_point = np.array(point) - reference_point
    rotated_translated_point = np.dot(rotation_matrix, translated_point)
    rotated_point = rotated_translated_point + reference_point
    
    return rotated_point

# ##################################################################################################################
# ###################################### ITERATING THROUGH THE TREE STRUCTURE ######################################
# ##################################################################################################################


# ##################################################################################################################
# ################################################ USEFUL FUNCTIONS ################################################
# #################################### Euclidean distance & Gradient Computation ###################################
# ##################################################################################################################
@jit(nopython=True)  # Set "nopython" mode for best performance, equivalent to @njit
def calculate_distance(p1, p2):
    """

    @param p1:
    @param p2:
    @return:
    """
    tmp_sum = 0


    dimensions = p1.shape[0]
    for i in range(0, dimensions):
        tmp_sum += (p1[i] - p2[i]) ** 2

    result = math.sqrt(tmp_sum)
    
    return result


def compute_gradient(init_fitness_score, target_fitness_score, init_coordinates, target_coordinates):
    distance_between_points = calculate_distance(init_coordinates, target_coordinates)
    if distance_between_points > 0 and init_fitness_score is not None and target_fitness_score is not None:
        gradient = (init_fitness_score - target_fitness_score) / distance_between_points
        # gradient = abs(init_fitness_score - target_fitness_score) / distance_between_points
    else:
        gradient = None
        # gradient = - np.float('Inf')
    return gradient


# ##################################################################################################################
# ############################################ SAVING TO FILE FUNCTIONS ############################################
# ##################################### Loading pickle & write & visualizations ####################################
# ##################################################################################################################
def parse_data_from_output_files(filename):
    a_file = open(filename, "rb")
    data = pickle.load(a_file)

    return data


def write_final_output_results():
    if constants.dataset_type in ["MPB"]:
        write_final_output_results_for_MPB()


def write_final_output_results_for_MPB():
    # Write solution to output file
    output_file = os.path.join(constants.output_folder, "_".join(["FINAL", constants.dataset_type + ".txt"]))
    f = open(output_file, "a")
    f.write(";".join(["AVG Offline Error", "STD Offline Error", "AVG Current Error", "STD Current Error"]) + "\n")
    avg_offline_errors = []
    std_offline_errors = []
    avg_current_errors = []
    std_current_errors = []
    for random_seed in constants.random_seeds:
        filename = os.path.join(constants.output_folder, str(random_seed) + '_execution_data.pkl')
        data = parse_data_from_output_files(filename)

        if constants.ils_visualization:
            utils_vis.output_visualization_hyperspheres(data, random_seed)
        utils_vis.output_visualization(random_seed, data["offlineError"], data["currentError"])

        LOGGER.critical("Offline Error - seed: " + str(random_seed))
        offline_error_per_period = [min(data["offlineError"][idx * data["period"]:(idx + 1) * data["period"]]) for idx
                                    in range(int(constants.datasets[constants.dataset_type]["max_evaluations"] // data["period"]))]
        avg_offline_errors.append(offline_error_per_period[-1])
        LOGGER.critical("AVG: " + str(avg_offline_errors[-1]))
        std_offline_errors.append(np.std(offline_error_per_period))
        LOGGER.critical("STD: " + str(std_offline_errors[-1]))

        LOGGER.critical("Current Error - seed: " + str(random_seed))
        current_error_per_period = [min(data["currentError"][idx * data["period"]:(idx + 1) * data["period"]]) for idx
                                    in range(int(constants.datasets[constants.dataset_type]["max_evaluations"] // data["period"]))]
        avg_current_errors.append(np.mean(current_error_per_period))
        LOGGER.critical("AVG: " + str(avg_current_errors[-1]))
        std_current_errors.append(np.std(current_error_per_period))
        LOGGER.critical("STD: " + str(std_current_errors[-1]))

        f.write(";".join([str(avg_offline_errors[-1]), str(std_offline_errors[-1]), str(avg_current_errors[-1]),
                          str(std_current_errors[-1])]) + "\n")

    LOGGER.critical("FINAL Offline Error")
    LOGGER.critical("AVG: " + str(np.mean(avg_offline_errors)))
    LOGGER.critical("STD: " + str(np.mean(std_offline_errors)))
    std_error_offline_errors = np.mean(std_offline_errors) / np.sqrt(len(std_offline_errors))
    LOGGER.critical("STD ERROR: " + str(std_error_offline_errors))

    LOGGER.critical("FINAL Current Error")
    LOGGER.critical("AVG: " + str(np.mean(avg_current_errors)))
    LOGGER.critical("STD: " + str(np.mean(std_current_errors)))
    std_error_current_errors = np.mean(std_current_errors) / np.sqrt(len(std_current_errors))
    LOGGER.critical("STD ERROR: " + str(std_error_current_errors))

    f.write("\n")
    f.write(";".join(
        ["AVG Offline Error", "STD Offline Error", "STD ERROR Offline Error", "AVG Current Error", "STD Current Error",
         "STD ERROR Current Error"]) + "\n")
    f.write(";".join(
        [str(np.mean(avg_offline_errors)), str(np.mean(std_offline_errors)), str(std_error_offline_errors),
         str(np.mean(avg_current_errors)), str(np.mean(std_current_errors)), str(std_error_current_errors)]) + "\n")
    f.close()
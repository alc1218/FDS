"""
Implementation of the FDS (Fractal Decomposition-based line Search Algorithm) as presented in
   *Llanza, Arcadi and Shvai, Nadiya and Nakib, Amir 2025, "FDS: Fractal Decomposition based Direct Search Approach for Continuous Dynamic Optimization"*
"""

import os
import pickle
import logging
import time
from datetime import timedelta

import FDS.FDS as FDS
import FDS.Detector as Detector
from FDS import constants
from FDS.dataset import datasets
from multiprocessing import Process
from FDS import utils

logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(constants.log_level)


def main(random_seed, output_folder, verbose=True):
    """

    @param random_seed: List of random seeds to operate on
    @param verbose:
    @return:
    """

    dataset = datasets.Dataset(random_seed, output_folder)
    fds = FDS.FDS(dataset)
    detector = Detector.Detector()

    is_fds_finished_count = 0
    while dataset.number_of_evaluations < constants.datasets[constants.dataset_type]["max_evaluations"]:

        LOGGER.debug(f"select_next_node_to_be_explored")
        st = time.time()
        is_landscape_changed, are_max_evaluations_reached = fds.select_next_node_to_be_explored()
        end = time.time()
        LOGGER.debug(f"Selection node time: {end - st}")
        if are_max_evaluations_reached:
                continue

        if not is_landscape_changed:
            LOGGER.debug(f"main_intensification")
            st = time.time()
            is_landscape_changed, are_max_evaluations_reached = fds.main_intensification()
            end = time.time()
            LOGGER.debug(f"Intensification node time: {end - st}")
            if are_max_evaluations_reached:
                continue

            if not is_landscape_changed:
                LOGGER.debug(f"decomposition_hypersphere")
                st = time.time()
                fds.check_if_all_FDS_structure_has_finished()
                if fds.is_fds_finished:
                    # Extend one level of hyperspheres
                    hyperspheres_to_decompose = fds.gather_hyperspheres(level=constants.k_levels_min)
                    constants.k_levels_min += 1
                    for hypersphere in hyperspheres_to_decompose:
                        fds.current_hypersphere = hypersphere
                        is_landscape_changed, are_max_evaluations_reached = fds.decomposition_hypersphere()
                end = time.time()
                LOGGER.debug(f"Decomposition node time: {end - st}")

        if is_landscape_changed:
            LOGGER.debug(f"check_if_landscape_has_changed")
            fds, is_landscape_changed = detector.check_if_landscape_has_changed(fds, dataset)

    # Close file
    fds.dataset.f.close()

    LOGGER.info("is_fds_finished_count: " + str(is_fds_finished_count))
    LOGGER.info("check_detector_points: " + str(detector.check_detector_points) + " (" + str(detector.check_detector_points / dataset.number_of_evaluations) + ")")

    if constants.record_pickle:
        dataset.data_to_be_stored["NDIM"] = dataset.number_of_dimensions
        # a_file = open(os.path.join(constants.output_folder, str(random_seed) + '_execution_data.pkl'), "wb")
        a_file = open(os.path.join(output_folder, str(random_seed) + '_execution_data.pkl'), "wb")
        pickle.dump(dataset.data_to_be_stored, a_file)
        a_file.close()

    if constants.is_visualization:
        dataset.out.release()


if __name__ == '__main__':
    LOGGER.info(f"HELLO WORLD")
    start_tp = time.time()

    # LOGGER.info(f"Output folder initialization")
    os.makedirs(constants.root_output_folder, exist_ok=True)
    os.makedirs(os.path.join(constants.root_output_folder, constants.dataset_type), exist_ok=True)
    os.makedirs(os.path.join(constants.root_output_folder, constants.dataset_type, constants.datetime_str), exist_ok=True)

    # Save parameters to output file
    output_file = os.path.join(constants.output_folder, "PARAMETERS.txt")
    f = open(output_file, "a")
    f.write(";".join(["Parameter", "Value"]) + "\n")
    f.write(";".join(["DIMENSIONS", str(constants.datasets[constants.dataset_type]["dimensions"])]) + "\n")
    f.write(";".join(["SEVERITY", str(constants.datasets[constants.dataset_type]["move_severity"])]) + "\n")
    f.write(";".join(["FREQUENCY", str(constants.datasets[constants.dataset_type]["period"])]) + "\n")
    f.write(";".join(["PEAKS", str(constants.datasets[constants.dataset_type]["npeaks"])]) + "\n")
    f.write(";".join(["HYPERSPHERE_LEVELS", str(constants.init_k_levels_min)]) + "\n")
    f.write(";".join(["GAMMA_MIN", str(constants.GAMMA_MIN)]) + "\n")
    f.write(";".join(["GAMMA_DECREASE_STEP", str(constants.GAMMA_DECREASE_STEP)]) + "\n")
    f.write(";".join(["STEP_SIZE", str(constants.step_size)]) + "\n")

    f.close()

    LOGGER.info(f"Start FDS")
    if len(constants.random_seeds) > 1:

        counter = 0
        while counter * constants.max_process_batch < len(constants.random_seeds):
            processes = []
            random_seeds = constants.random_seeds[counter * constants.max_process_batch:(counter+1) * constants.max_process_batch]

            LOGGER.info(f"random_seeds: {random_seeds}")
            for random_seed in random_seeds:
                processes.append(Process(target=main, args=(random_seed, constants.output_folder)))
                processes[-1].start()

            for process in processes:
                process.join()
            counter += 1
    else:
        main(constants.random_seeds[0], constants.output_folder)

    LOGGER.info(f"Starting to generate graphs & overall results")
    utils.write_final_output_results()
    end_tp = time.time()
    
    # Calculate the difference
    time_difference = end_tp - start_tp
    
    # Convert the difference to a timedelta object
    timedelta_obj = timedelta(seconds=time_difference)
    LOGGER.info(f"Total time: {timedelta_obj}")

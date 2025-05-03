from FDS import constants

import numpy as np
import io
import ast

import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from FDS import utils
import collections
import random
from tqdm import tqdm


logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(constants.log_level)


def visualize_hypersphere_centers_decomposition(fds, random_seed):
    hyperspheres = fds.gather_hyperspheres(level=-1)

    fig = plt.figure(figsize=(30, 30))
    # plt.title(f"fds decomposition ({constants.init_k_levels_min} levels)")
    ax = plt.gca()
    ax.set_xlim([-0.5, 1.5])
    ax.set_ylim([-0.5, 1.5])

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    ax.plot([0, 0], [1, 0], color="black", linewidth=5)
    ax.plot([0, 1], [0, 0], color="black", linewidth=5)
    ax.plot([1, 0], [1, 1], color="black", linewidth=5)
    ax.plot([1, 1], [0, 1], color="black", linewidth=5)

    color_palette = ['Grey', 'Red', 'Purple', 'Blue', 'Green', 'Orange']
    for hypersphere in hyperspheres:
        color = color_palette[hypersphere.current_hypersphere_level]
        coordinates = hypersphere.center

        ax.plot(coordinates[0], coordinates[1], '*', color=color, markersize=50)
        circle = plt.Circle((coordinates[0], coordinates[1]), hypersphere.radius, color=color, fill=False, linewidth=5)
        inflated_circle = plt.Circle((coordinates[0], coordinates[1]), hypersphere.radius * constants.INFLATED_RATE, color=color, fill=False, linestyle=(0, (5, 20)), linewidth=5)

        ax.add_patch(circle)
        ax.add_patch(inflated_circle)

    fig.tight_layout()
    plt.savefig(os.path.join(constants.output_folder, f"{random_seed}_FDS_decomposition_{hypersphere.current_hypersphere_level}_levels"))


def visualize_hypersphere_centers_per_peak(fds, dataset, random_seed):
    hyperspheres = fds.gather_hyperspheres(level=-1)
    hyperspheres_coordinates = [hypersphere.center for hypersphere in hyperspheres]

    fitnesses = []
    peaks_attraction = []
    for hypersphere_coordinates in hyperspheres_coordinates:
        fitness, peak_attraction = dataset.evaluate_fitness_dummy(hypersphere_coordinates)
        fitnesses.append(fitness)
        peaks_attraction.append(peak_attraction)

    _ = plt.hist(peaks_attraction, bins=10)
    plt.title(f"Histogram of peaks attraction per hypersphere centers for random_seed: {random_seed}")
    plt.savefig(os.path.join(constants.output_folder, f"{random_seed}_histogram_of_peaks_attraction_per_hypersphere_centers"))
    plt.clf()

    occurrences = collections.Counter(peaks_attraction)
    LOGGER.info(f"Peaks that can be seen based on the initial decomposition: {occurrences}")

    maximums_coordinates = [maximum[1] for maximum in dataset.mpb.maximums()]
    distances = []
    for maximum_coordinates in maximums_coordinates:
        distances_local = []
        for hypersphere_coordinates in hyperspheres_coordinates:
            distances_local.append(utils.calculate_distance(np.asarray(maximum_coordinates) / 100, np.asarray(hypersphere_coordinates)))
        distances.append(distances_local)
    all_available_peaks = np.unique(np.argmin(distances, axis=0))

    _ = plt.hist(np.argmin(distances, axis=0), bins=10)
    plt.title(f"Histogram of closest peak per hypersphere centers for random_seed: {random_seed}")
    plt.savefig(os.path.join(constants.output_folder, f"{random_seed}_histogram_of_closest_peak_hypersphere_centers"))
    plt.clf()

    LOGGER.info(f"all_available_peaks: {all_available_peaks}")


def compute_zone_percentage_peak_attraction_influence(dataset, random_seed):

    random.seed(random_seed)
    number_of_samples = 100000 + 10
    fitnesses = []
    peaks_attraction = []
    for sample in tqdm(range(number_of_samples)):
        hypersphere_coordinates = []
        for d in range(5):
            hypersphere_coordinates.append(random.uniform(0.0, 1.0))
        fitness, peak_attraction = dataset.evaluate_fitness_dummy(np.asarray(hypersphere_coordinates))

        fitnesses.append(fitness)
        peaks_attraction.append(peak_attraction)

    fitnesses = fitnesses[10:]
    peaks_attraction = peaks_attraction[10:]

    occurrences = collections.Counter(peaks_attraction)
    LOGGER.info(f"Peaks that can be seen based on 100k random uniform samples on the search space: {occurrences}")


def compute_statistics_over_hypersphere_centers_searches(data):
    stats_dict = {}
    for sample in np.unique(data["search_step"]):
        stats_dict[sample] = {
            "times": [],
            "evaluation_idx": []
        }

    counter = 0
    step_type = data["search_step"][0]
    for evaluation_idx, step in enumerate(data["search_step"]):
        if step_type == step:
            counter += 1
        else:
            stats_dict[step_type]["times"].append(counter)
            stats_dict[step_type]["evaluation_idx"].append(evaluation_idx)

            step_type = step
            counter = 1

    return stats_dict


def compute_histogram(stats_dict, random_seed):
    for stats_dict_key in stats_dict.keys():
        _ = plt.hist(stats_dict[stats_dict_key]["times"], bins='auto')
        # stats_dict[stats_dict_key]["times"]
        # _ = plt.hist(stats_dict[stats_dict_key]["times"], bins=len(stats_dict[stats_dict_key]["times"]) // 500)
        plt.title(f"Histogram of {stats_dict_key} for random_seed: {random_seed}")

        plt.savefig(os.path.join(constants.output_folder, f"{random_seed}_histogram_of_{stats_dict_key}"))
        plt.clf()


def compute_module_percentage_utilization(stats_dict):
    total_evaluations_all_modules = 0
    for stats_dict_key in stats_dict.keys():
        total_evaluations = np.sum(stats_dict[stats_dict_key]["times"])
        total_evaluations_all_modules += total_evaluations

    module_percentage = ""
    for stats_dict_key in stats_dict.keys():
        total_evaluations = np.sum(stats_dict[stats_dict_key]["times"])

        module_percentage += f" {stats_dict_key} {total_evaluations / total_evaluations_all_modules}"

    LOGGER.critical("Percentage split spend in different modules:")
    LOGGER.critical(module_percentage[1:])


def collect_ILS_data(data):
    ils_data = {
        "fitness_scores": [],
        "color_types": [],
        "evaluations": [],
        "hypersphere_IDs": []
    }

    fitness_scores = []
    color_types = []
    evaluations = []
    hypersphere_IDs = None
    previous_sample = None
    for idx_sample, sample in enumerate(data["hypersphere_IDs"]):
        if data["search_step"][idx_sample] not in ["H", "Detector", "ILS_center_point"]:
            if previous_sample is None:
                previous_sample = sample

            if previous_sample != sample:
                ils_data["fitness_scores"].append(fitness_scores)
                ils_data["color_types"].append(color_types)
                ils_data["evaluations"].append(evaluations)
                ils_data["hypersphere_IDs"].append(hypersphere_IDs)

                fitness_scores = []
                color_types = []
                evaluations = []
                hypersphere_IDs = None

                previous_sample = sample

            fitness_scores.append(data["fitness_score"][idx_sample])
            color_types.append(data["peaks_attraction"][idx_sample])
            evaluations.append(idx_sample)
            if hypersphere_IDs is None:
                hypersphere_IDs = sample

    return ils_data


def create_graph_performance_per_hypersphere(fitness_scores, color_types, evaluations, hypersphere_IDs, random_seed, sufix_type):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    fig = plt.figure(figsize=(30, 30))

    for counter, (fitness_scores_sample, color_types_sample, evaluations_sample, hypersphere_IDs_sample) in enumerate(zip(fitness_scores, color_types, evaluations, hypersphere_IDs)):
        plt.subplot(10, 10, counter + 1)
        plt.title(f"Hypershere ID: {hypersphere_IDs_sample}")
        ax = plt.gca()
        ax.set_ylim([-200, 75])

        for fitness_score, color_type, idx_sample in zip(fitness_scores_sample, color_types_sample, evaluations_sample):
            plt.scatter(idx_sample, fitness_score, color=colors[color_type])

    fig.tight_layout()
    plt.savefig(os.path.join(constants.output_folder, f"{random_seed}_hyperspheres_performance{sufix_type}"))


def output_visualization_hyperspheres(data, random_seed):
    # Compute statistics over hypersphere centers searches
    stats_dict = compute_statistics_over_hypersphere_centers_searches(data)

    # Create histogram
    compute_histogram(stats_dict, random_seed)

    # Compute module percentage utilization
    compute_module_percentage_utilization(stats_dict)

    # Create graphs performance per hypersphere
    ils_data = collect_ILS_data(data)

    # Post process data
    ils_data_lists = list(ils_data.items())
    fitness_scores = ils_data_lists[0][1]
    color_types = ils_data_lists[1][1]
    evaluations = ils_data_lists[2][1]
    hypersphere_IDs = ils_data_lists[3][1]

    create_graph_performance_per_hypersphere(fitness_scores[:100], color_types[:100], evaluations[:100], hypersphere_IDs[:100], random_seed, sufix_type="_1st_period")

    # Post process data ==> Sort data by worst values
    total_number_of_evaluations = [len(evaluation_sample) for evaluation_sample in evaluations]

    fitness_scores = [x for _, x in sorted(zip(total_number_of_evaluations, fitness_scores))]
    color_types = [x for _, x in sorted(zip(total_number_of_evaluations, color_types))]
    evaluations = [x for _, x in sorted(zip(total_number_of_evaluations, evaluations))]
    hypersphere_IDs = [x for _, x in sorted(zip(total_number_of_evaluations, hypersphere_IDs))]

    # LOGGER.info(f"BEST evaluations: {total_number_of_evaluations}")
    create_graph_performance_per_hypersphere(fitness_scores[:100], color_types[:100], evaluations[:100], hypersphere_IDs[:100], random_seed, sufix_type="_best")

    # Post process data ==> Sort data by worst values
    total_number_of_evaluations = [len(evaluation_sample) for evaluation_sample in evaluations]

    fitness_scores = [x for _, x in sorted(zip(total_number_of_evaluations, fitness_scores), reverse=True)]
    color_types = [x for _, x in sorted(zip(total_number_of_evaluations, color_types), reverse=True)]
    evaluations = [x for _, x in sorted(zip(total_number_of_evaluations, evaluations), reverse=True)]
    hypersphere_IDs = [x for _, x in sorted(zip(total_number_of_evaluations, hypersphere_IDs), reverse=True)]

    # data["hypersphere_IDs"][evaluations[0][0] : evaluations[0][-1]]
    # import ipdb; ipdb.set_trace()

    # LOGGER.info(f"WORST evaluations: {total_number_of_evaluations}")
    create_graph_performance_per_hypersphere(fitness_scores[:100], color_types[:100], evaluations[:100], hypersphere_IDs[:100], random_seed, sufix_type="_worst")


def output_visualization(random_seed, offline_error, current_error):
    """
    Visualization function
    @param random_seed:
    @param offline_error:
    @param current_error:
    @return:
    """
    # create_visual_plots()
    fig, ax = plt.subplots()
    ax.plot(offline_error[:int(constants.datasets[constants.dataset_type]["max_evaluations"])], '-b', label='Offline Error')
    ax.plot(current_error[:int(constants.datasets[constants.dataset_type]["max_evaluations"])], '--r', label='Current Error')
    plt.xlabel('Function Evaluations')
    plt.ylabel('Offline Error and Current Error')
    plt.gca().set_ylim(bottom=0)
    leg = ax.legend()
    # plt.show()
    plt.savefig(os.path.join(constants.output_folder, str(random_seed) + '_mpb_fds_v1.png'))

    plt.close(fig)


def create_figure(DPI):
    fig = plt.figure(figsize=(8, 8), dpi=DPI)

    plt.ylim(-0.5, 1.5)
    plt.xlim(-0.5, 1.5)

    plt.xlabel("1st Dimension")
    plt.ylabel("2nd Dimension")

    # Limit definition
    plt.plot([0, 1], [0, 0], 'm-', label="Search Space limit")
    plt.plot([0, 1], [1, 1], 'm-')
    plt.plot([0, 0], [0, 1], 'm-')
    plt.plot([1, 1], [0, 1], 'm-')

    return fig


def create_contours(toolbox):
    xs = np.linspace(0, 1, 50)
    ys = np.linspace(0, 1, 50)

    X, Y = np.meshgrid(xs, ys)

    Z = np.zeros([X.shape[0], X.shape[1]])
    for x_idx, x in enumerate(xs):
        for y_idx, y in enumerate(ys):
            Z[x_idx, y_idx] = toolbox.evaluate([x, y])[0]

    plt.contourf(X, Y, Z, 20, cmap='RdGy')
    plt.colorbar()


def add_maximums_data(maximums):
    # Add maximums
    x = [sample[1][0] / 100 for sample in maximums]
    y = [sample[1][1] / 100 for sample in maximums]

    plt.scatter(x, y, marker="P", color='black', alpha=0.5, label='Local Optima')


def add_hyperspheres_centers_data(hyperspheres_centers):

    # Add hypersphere centers (Non explored solutions)
    if hyperspheres_centers is not None:
        x_exploited = []
        y_exploited = []
        x_NOT_exploited = []
        y_NOT_exploited = []
        for hypersphere_sample in hyperspheres_centers:
            if hypersphere_sample.is_hypersphere_exploited:
                x_exploited.append(hypersphere_sample.center[0])
                y_exploited.append(hypersphere_sample.center[1])
            else:
                x_NOT_exploited.append(hypersphere_sample.center[0])
                y_NOT_exploited.append(hypersphere_sample.center[1])

        if x_exploited:
            plt.scatter(x_exploited, y_exploited, marker="D", color='green', alpha=0.5, label='Explored - Hypersphere centers')
        if x_NOT_exploited:
            plt.scatter(x_NOT_exploited, y_NOT_exploited, marker="D", color='red', alpha=0.5, label='Non Explored - Hypersphere centers')


def add_searched_solutions_data(searched_solutions):
    # Add searched_solutions
    x = []
    y = []
    for sample in searched_solutions.keys():
        # fitness_value = searched_solutions[sample]["fitness_value"]
        sample = ast.literal_eval(sample)
        x.append(sample[0])
        y.append(sample[1])

    plt.scatter(x, y, color='blue', alpha=0.1, label='Searched solutions')


def get_frame_from_plt(fig, DPI):
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw', dpi=DPI)
    io_buf.seek(0)
    frame = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                       newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))

    io_buf.close()

    return frame


def create_frame(searched_solutions, maximums, hyperspheres_centers, toolbox):
    # Create figure
    DPI = 64
    fig = create_figure(DPI)

    # Create contour lines
    # create_contours(toolbox)

    # Add all data points
    add_hyperspheres_centers_data(hyperspheres_centers)
    add_searched_solutions_data(searched_solutions)
    add_maximums_data(maximums)

    # Add legend of drawn data
    plt.legend()

    # Get stream of bytes from PLT to NP
    frame = get_frame_from_plt(fig, DPI)

    # Select only RGB channels
    frame = frame[:, :, 0:3]

    return frame

from FDS import utils


class UtilsDecomposition:
    # ##################################################################################################################
    # ###################################################### UTILS #####################################################
    # ################################################ GATHER POPULATION ###############################################
    # ##################################################################################################################
    def gather_population(self):
        # Iterate through all the leaf nodes
        stack = [self.initial_hypersphere]
        leaf_nodes = []
        while stack:
            hypersphere_instance = stack.pop()
            if hypersphere_instance.hypersphere_children:
                stack += hypersphere_instance.hypersphere_children
            else:
                if hypersphere_instance.fitness != float('Inf'):
                    leaf_nodes.append(hypersphere_instance)

        return leaf_nodes

    def gather_local_minimum_hyperspheres(self):
        # Iterate through all the leaf nodes
        stack = [self.initial_hypersphere]
        local_minimums = []
        while stack:
            hypersphere_instance = stack.pop()
            if hypersphere_instance.hypersphere_children:
                stack += hypersphere_instance.hypersphere_children
            else:
                if hypersphere_instance.is_local_minimum:
                    local_minimums.append(hypersphere_instance)

        return local_minimums

    def gather_hyperspheres(self, level):
        """
        Gather all hyperspheres from given level. If level is -1 it selects all the hyperspheres
        @param level:
        @return:
        """
        # Iterate through all the leaf nodes
        stack = [self.initial_hypersphere]
        population = []
        while stack:
            hypersphere_instance = stack.pop()
            stack += hypersphere_instance.hypersphere_children
            if level in [hypersphere_instance.current_hypersphere_level, -1]:
                population.append(hypersphere_instance)
        return population

    def get_closest_points(self, fitness_coordinates, number_of_points):
        # Iterate through all the hyperspheres
        stack = [self.initial_hypersphere]
        hyperspheres = []
        while stack:
            hypersphere_instance = stack.pop()
            if hypersphere_instance.hypersphere_children:
                stack += hypersphere_instance.hypersphere_children
                hyperspheres.append(hypersphere_instance)

        distances = [utils.calculate_distance(fitness_coordinates, hypersphere.fitness_coordinates) for hypersphere in hyperspheres if max(hypersphere.fitness_coordinates) < self.dataset.upper_bound_norm and min(hypersphere.fitness_coordinates) > self.dataset.lower_bound_norm]
        hyperspheres = [hypersphere_instance for _, hypersphere_instance in sorted(zip(distances, hyperspheres), key=lambda pair: pair[0])]

        hypersphere_instance = hyperspheres[:number_of_points]

        return hypersphere_instance
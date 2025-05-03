import copy as copy
import logging
from FDS import constants

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(constants.log_level)


class Hypersphere:

	fitness_coordinates = []
	fitness = None

	best_ratio = None

	class_counter = 1

	def __init__(self, tmp_center, tmp_radius, current_hypersphere_level, hypersphere_parent):
		"""

		@param tmp_center:
		@param tmp_radius:
		"""
		Hypersphere.class_counter += 1
		self.center = copy.copy(tmp_center)
		self.radius = tmp_radius
		self.current_hypersphere_level = current_hypersphere_level
		self.hypersphere_parent = hypersphere_parent
		self.hypersphere_children = []
		self.is_hypersphere_exploited = False
		self.is_local_minimum = False

		self.historic_solutions = []
		self.historic_fitness = []
		self.historic_evaluation_step = []

		self.id = "_".join([str(self.current_hypersphere_level), str(self.class_counter)])
		# LOGGER.info("HyperSphere created dimension,tmp_center, tmp_radius")

	def inflate_hypersphere(self):
		"""

		@return:
		"""
		self.radius = self.radius * constants.INFLATED_RATE

	def __str__(self):
		"""
		Method that allows to overwrite the print description of an object
		@return: 
		"""
		for i in range(0, constants.datasets[constants.dataset_type]["dimensions"]):
			print("Dim["), 			
			print(i), 			
			print("] : "), 			
			print(self.center[i])
		print(self.fitness)	
		print(self.best_ratio)
		return ""

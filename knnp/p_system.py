"""
This module implements the main functionality of knnp.

Author: Khushiyant
(Cite) Paper Basis: https://doi.org/10.1016/j.tcs.2020.01.001
"""

__author__ = "Khushiyant"
__email__ = "khushiyant2002@gmail.com"


import numpy as np
from typing import List, Tuple


class kNN_P:
    def __init__(self, n: int, d: int, q: int, m: int, k: int, maxstep: int, w=0.5, c1=1.5, c2=1.5):
        """
        Initializes the kNN-P algorithm.

        Parameters:
        -----------
        n : int
            The number of training samples. It represents the size of your training dataset, i.e., the number of `data points` used for training the ``kNN-P algorithm`.
        d : int
            The number of attributes or features in your data. It indicates the dimensionality of each data point in your dataset.
        q : int
            The number of test samples. This parameter specifies how many data points you want to make predictions for using the `kNN-P algorithm`.
        m : int
            The number of objects in each cell. In the context of the algorithm, a `cell` contains multiple objects, and this parameter specifies the number of objects per cell.
        k : int
            The number of nearest neighbors to find. This is a key parameter of the k-nearest neighbors algorithm, and it determines how many nearest neighbors will be considered for each test sample.
        maxstep : int
            The maximum number of iterations or steps for the optimization process. It controls how many iterations the algorithm will perform to refine the nearest neighbors.
        w : float, optional
            This is a parameter for the `PSO (Particle Swarm Optimization)` component of the algorithm. It controls the `inertia weight`, affecting the balance between exploration and exploitation during optimization. Default is 0.5.
        c1 : float, optional
            Another parameter for the PSO component, it controls the cognitive component of the PSO algorithm. Default is 1.5.
        c2 : float, optional
            A parameter for the PSO component, controlling the social component of the PSO algorithm. Default is 1.5.
        """
        self.n = n
        self.d = d
        self.q = q
        self.m = m
        self.k = k
        self.maxstep = maxstep
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.cells = [np.random.randint(
            0, self.n, size=(self.m, self.k)) for _ in range(q)]
        self.environment = np.zeros((q, k), dtype=int)

    def _evaluate_fitness(self, object, test_sample) -> float:
        """
        Evaluates the fitness of an object for a given test sample by calculating the sum of distances between the training samples indexed by the object and the test sample.

        Parameters:
        -----------
        object : numpy array
            The object to evaluate.
        test_sample : numpy array
            The test sample to evaluate the object on.

        Returns:
        --------
        fitness : float
            The fitness of the object for the given test sample.
        """
        distances = [self._calculate_distance(
            self.training_data[i - 1], test_sample) for i in object]
        fitness = sum(distances)
        return fitness

    def _calculate_distance(self, point1, point2) -> float:
        """
        Calculates the Euclidean distance between two data points, `point1` and `point2`.

        Parameters:
        -----------
        point1 : numpy array
            The first point.
        point2 : numpy array
            The second point.

        Returns:
        --------
        distance : float
            The Euclidean distance between the two points.
        """
        return np.linalg.norm(point1 - point2)

    def _communicate_best_object(self, environment, cell_object, test_sample) -> None:
        """
        Updates the best objects in the environment based on the fitness of objects in the cells for a specific test sample.

        Parameters:
        -----------
        environment : numpy array
            The environment to communicate the best object to.
        cell_object : numpy array
            The cell object to evaluate.
        test_sample : numpy array
            The test sample to evaluate the cell object on.
        """
        cell_fitness = self._evaluate_fitness(cell_object, test_sample)
        environment_fitness = self._evaluate_fitness(environment, test_sample)

        if cell_fitness < environment_fitness:
            for j in range(self.k):
                environment[j] = cell_object[np.argmin(cell_fitness)][j]

    def _evolve_objects(self, objects, best_objects, environment) -> np.ndarray:
        """
        Implements the `Particle Swarm Optimization (PSO)` rule to update the objects in each cell during optimization.

        Parameters:
        -----------
        objects : numpy array
            The objects to evolve.
        best_objects : numpy array
            The personal best positions of the objects.
        environment : numpy array
            The global best position of the objects.

        Returns:
        --------
        objects_new : numpy array
            The evolved objects.
        """
        r1 = np.random.random()
        r2 = np.random.random()

        velocity = self.w * objects + self.c1 * r1 * \
            (best_objects - objects) + self.c2 * r2 * (environment - objects)
        objects_new = np.round(objects + velocity)

        # Ensure that the values stay within the bounds of training data indices
        objects_new = np.clip(objects_new, 1, self.n)

        return objects_new

    def fit(self, training_data: np.ndarray, class_labels) -> None:
        """
        The main function for training the kNN-P classifier. It iteratively optimizes the objects in the cells and updates the best objects in the environment.

        Parameters:
        -----------
        training_data : numpy array
            The training data to fit the algorithm to.
        """
        if training_data.shape[0] != class_labels.shape[0]:
            raise ValueError(
                "The number of training samples and class labels must be equal.")

        self.training_data = training_data

        for _ in range(self.maxstep):
            for i in range(self.q):
                for j in range(self.m):
                    # Evaluate fitness
                    fitness = [self._evaluate_fitness(
                        cell, self.training_data[i]) for cell in self.cells[i]]
                    best_object = self.cells[i][np.argmin(fitness)]
                    self.cells[i][j] = self._evolve_objects(
                        self.cells[i][j], best_object, self.environment[i])

            # Communication step
            for i in range(self.q):
                self._communicate_best_object(
                    self.environment[i], self.cells[i], self.training_data[i])

    def predict(self, test_data: np.ndarray) -> List[Tuple[List[int], List[float]]]:
        """
        Predicts the k nearest neighbors for the test samples using the trained kNN-P classifier. It returns a list of tuples, each containing the indices of the nearest neighbors and the distances.

        Parameters:
        -----------
        test_data : numpy array
            The test data to predict the k nearest neighbors for.

        Returns:
        --------
        predictions : list of tuples
            The k nearest neighbors for each query point.
        """
        self.test_data = test_data
        predictions = []
        for i in range(self.q):
            neighbors_indices = self.environment[i]
            distances = [self._calculate_distance(
                self.test_data[i], self.test_data[index - 1]) for index in neighbors_indices]
            predictions.append((neighbors_indices, distances))
        return predictions

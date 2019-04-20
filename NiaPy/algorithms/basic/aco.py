# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy, no-self-use, line-too-long, arguments-differ, bad-continuation
import logging

from numpy import array, inf, where

from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['AntColonyOptimization']

class AntColonyOptimization(Algorithm):
    r"""Implementation of Ant Colony Optimization algorithm.

        Algorithm:
            Ant colony optimization

        Date:
            2019

        Author:
            Ivan Kozulić

        License:
            MIT

        Reference paper:
            M. Duran Toksari. "Ant colony optimization for finding the global minimum". Applied Mathematics and Computation Volume 176, Issue 1, 1 May 2006, Pages 308-316.

        Attributes:
            Name (List[str]): List of strings representing algorithm names.

        See Also:
            * :class:`NiaPy.algorithms.Algorithm`
        """
    Name = ['AntColonyOptimization', 'ACO']

    @staticmethod
    def typeParameters():
        r"""Get dictionary with functions for checking values of parameters.

        Returns:
            Dict[str, Callable]:
                * NP (Callable[[int], bool]): Checks if number of individuals in population parameter has a proper value.
                * alphaMin (Callable[[Union[int, float]], bool]): Checks if alpha minimum parameter has a proper value.
                * alphaMax (Callable[[Union[int, float]], bool]): Checks if alpha maximum parameter has a proper value.
                * pheromone (Callable[[Union[int, float]], bool]): Checks if pheromone parameter has a proper value.
                * evaporation (Callable[[Union[int, float]], bool]): Checks if evaporation parameter has a proper value.

        See Also:
            * :func:`NiaPy.algorithms.algorithm.Algorithm.typeParameters`
        """
        d = Algorithm.typeParameters()
        d.update({
            'NP': lambda x: isinstance(x, int) and x > 0,
            'alphaMin': lambda x: isinstance(x, (float, int)),
            'alphaMax': lambda x: isinstance(x, (float, int)),
            'jumpLength': lambda x: isinstance(x, float) and 0 < x < 1,
            'pheromone': lambda x: isinstance(x, (float, int)) and x >= 0,
            'evaporation': lambda x: isinstance(x, (float, int)) and 0 <= x < 1,
        })
        return d

    def setParameters(self, NP=25, alphaMin=-1.0, alphaMax=1.0, jumpLength=0.9, pheromone=0, evaporation=0.1, **ukwargs):
        r"""Set the parameters of the algorithm.

        Args:
            NP (Optional[int]): Population size
            alphaMin (Optional[float]): Alpha minimum parameter
            alphaMax (Optional[float]): Alpha maximum parameter
            jumpLength (Optional[float]): Jump length parameter
            pheromone (Optional[float]): Initial quantity of pheromone
            evaporation (Optional[float]): Evaporation factor
            **ukwargs: Additional arguments

        See Also:
            * :func:`NiaPy.algorithms.Algorithm.setParameters`
        """
        Algorithm.setParameters(self, NP=NP)
        self.alphaMin, self.alphaMax, self.jumpLength, self.pheromone, self.evaporation = alphaMin, alphaMax, jumpLength, pheromone, evaporation
        if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

    def determineDirectionOfMovement(self, best_solution):
        if (best_solution + (best_solution * 0.01)) <= best_solution:
            return 1
        return -1

    def updatePheromone(self, best_solution):
        self.pheromone = self.evaporation * self.pheromone
        self.pheromone = self.pheromone + (0.01 * best_solution)

    def repair(self, x, task):
        """Find limits."""
        ir = where(x > task.bcUpper())
        x[ir] = task.bcUpper()[ir]
        ir = where(x < task.bcLower())
        x[ir] = task.bcLower()[ir]
        return x

    def initPopulation(self, task):
        r"""Initialize population.

        Args:
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
                1. Initialized population.
                2. Initialized populations fitness/function values.
                3. Additional arguments:
                    * best_solution (float): Initial best solution.
                    * best_vector (numpy.ndarray): Initial best vector.
                    * direction_of_movement (int): Direction of ants movement.
                    * x_previous (float): Previous best solution.
                    * x_previous_vector (numpy.ndarray): Previous best vector.

        See Also:
            * :func:`NiaPy.algorithms.Algorithm.initPopulation`
        """
        pop, fpop, d = Algorithm.initPopulation(self, task)
        self.alphaMin = task.Lower
        self.alphaMax = task.Upper
        best_solution, best_vector = inf, None
        for i in range(self.NP):
            pop[i] = self.repair(pop[i], task)
            f = task.eval(pop[i])
            if f < best_solution:
                best_solution, best_vector = f, pop[i]
        direction_of_movement = self.determineDirectionOfMovement(best_solution)
        x_previous, x_previous_vector = best_solution, best_vector
        d.update({'best_solution': best_solution, 'best_vector': best_vector, 'direction_of_movement': direction_of_movement, 'x_previous': x_previous, 'x_previous_vector': x_previous_vector})
        return pop, fpop, d

    def runIteration(self, task, pop, fpop, xb, fxb, best_solution, best_vector, direction_of_movement, x_previous, x_previous_vector, **dparams):
        r"""Core function of Ant Colony Optimization.

        Args:
            task (Task): Optimization task.
            pop (numpy.ndarray): Current population.
            fpop (numpy.ndarray[float]): Current populations function/fitness values.
            xb (numpy.ndarray):
            fxb (float):
            best_solution (float): Initial best solution.
            best_vector (numpy.ndarray): Initial best vector.
            direction_of_movement (int): Direction of ants movement.
            x_previous (float): Previous best solution.
            x_previous_vector (numpy.ndarray): Previous best vector.
            **dparams (Dict[str, Any]): Additional arguments.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
                1. New population
                2. New population fitness/function values
                3. Additional arguments:
                    best_solution (float): Initial best solution.
                    best_vector (numpy.ndarray): Initial best vector.
                    direction_of_movement (int): Direction of ants movement.
                    x_previous (float): Previous best solution.
                    x_previous_vector (numpy.ndarray): Previous best vector.
        """
        for i, w in enumerate(pop):
            dx = self.uniform(self.alphaMin, self.alphaMax, task.D)
            if direction_of_movement == 1:
                pop[i] = array(x_previous_vector) + array(dx)
            elif direction_of_movement == -1:
                pop[i] = array(x_previous_vector) - array(dx)
        x_best_solution, x_best_vector = inf, None
        for i in range(self.NP):
            pop[i] = self.repair(pop[i], task)
            f = task.eval(pop[i])
            if f < x_best_solution:
                x_best_solution, x_best_vector = f, pop[i]
        if x_best_solution <= x_previous:
            best_solution, best_vector = x_best_solution, x_best_vector
        else:
            best_solution, best_vector = x_previous, x_previous_vector
        self.updatePheromone(x_best_solution)
        x_previous, x_previous_vector = x_best_solution, x_best_vector
        self.alphaMin, self.alphaMax = self.jumpLength * self.alphaMin, self.jumpLength * self.alphaMax
        for i, w in enumerate(pop):
            fpop[i] = task.eval(pop[i])
        return pop, fpop, {'best_solution': best_solution, 'best_vector': best_vector, 'direction_of_movement': direction_of_movement, 'x_previous': x_previous, 'x_previous_vector': x_previous_vector}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3

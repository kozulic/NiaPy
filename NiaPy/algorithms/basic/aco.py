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
            jumpLength (float): Jump length parameter.
            pheromone (float): Initial quantity of pheromone parameter.
            evaporation (float): Evaporation parameter.

        See Also:
            * :class:`NiaPy.algorithms.Algorithm`
        """
    Name = ['AntColonyOptimization', 'ACO']

    @staticmethod
    def algorithmInfo():
        return r"""
            Description: The ant colony optimization (ACO) algorithms are multi-agent systems in which the behaviour of each ant is inspired by the foraging behaviour of real ants to solve optimization problem.
            Authors: M. Duran Toksari
            Year: 2006
            Main reference: M. Duran Toksari. "Ant colony optimization for finding the global minimum". Applied Mathematics and Computation Volume 176, Issue 1, 1 May 2006, Pages 308-316.
        """

    @staticmethod
    def typeParameters():
        r"""Get dictionary with functions for checking values of parameters.

        Returns:
            Dict[str, Callable]:
                * jumpLength (Callable[[float], bool]): Checks if jump length parameter has a proper value.
                * pheromone (Callable[[Union[int, float]], bool]): Checks if pheromone parameter has a proper value.
                * evaporation (Callable[[Union[int, float]], bool]): Checks if evaporation parameter has a proper value.

        See Also:
            * :func:`NiaPy.algorithms.algorithm.Algorithm.typeParameters`
        """
        d = Algorithm.typeParameters()
        d.update({
            'jumpLength': lambda x: isinstance(x, float) and 0 < x < 1,
            'pheromone': lambda x: isinstance(x, (float, int)) and x >= 0,
            'evaporation': lambda x: isinstance(x, (float, int)) and 0 <= x < 1,
        })
        return d

    def setParameters(self, NP=25, jumpLength=0.9, pheromone=0, evaporation=0.1, **ukwargs):
        r"""Set the parameters of the algorithm.

        Args:
            NP (Optional[int]): Population size
            jumpLength (Optional[float]): Jump length parameter
            pheromone (Optional[float]): Initial quantity of pheromone
            evaporation (Optional[float]): Evaporation factor
            ukwargs (Dict[str, Any]): Additional arguments.

        See Also:
            * :func:`NiaPy.algorithms.Algorithm.setParameters`
        """
        Algorithm.setParameters(self, NP=NP, **ukwargs)
        self.jumpLength, self.pheromone, self.evaporation = jumpLength, pheromone, evaporation
        if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

    def determineDirectionOfMovement(self, best_solution):
        if (best_solution + (best_solution * 0.01)) <= best_solution:
            return 1
        return -1

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
                    * alphaMin (float): Minimum alpha.
                    * alphaMax (float): Maximum alpha.
                    * pheromone_quantity (float): Current quantity of pheromone.

        See Also:
            * :func:`NiaPy.algorithms.Algorithm.initPopulation`
        """
        pop, fpop, d = Algorithm.initPopulation(self, task)
        alphaMin, alphaMax = task.Lower, task.Upper
        pheromone_quantity = self.pheromone
        best_vector, best_solution = self.getBest(pop, fpop)
        direction_of_movement = self.determineDirectionOfMovement(best_solution)
        x_previous, x_previous_vector = best_solution, best_vector
        d.update({'best_solution': best_solution, 'best_vector': best_vector, 'direction_of_movement': direction_of_movement, 'x_previous': x_previous, 'x_previous_vector': x_previous_vector, 'alphaMin': alphaMin, 'alphaMax': alphaMax, 'pheromone_quantity': pheromone_quantity})
        return pop, fpop, d

    def runIteration(self, task, pop, fpop, xb, fxb, best_solution, best_vector, direction_of_movement, x_previous, x_previous_vector, alphaMin, alphaMax, pheromone_quantity, **dparams):
        r"""Core function of Ant Colony Optimization.

        Args:
            * task (Task): Optimization task.
            * pop (numpy.ndarray): Current population.
            * fpop (numpy.ndarray[float]): Current populations function/fitness values.
            * xb (numpy.ndarray):
            * fxb (float):
            * best_solution (float): Initial best solution.
            * best_vector (numpy.ndarray): Initial best vector.
            * direction_of_movement (int): Direction of ants movement.
            * x_previous (float): Previous best solution.
            * x_previous_vector (numpy.ndarray): Previous best vector.
            * alphaMin (float): Minimum alpha.
            * alphaMax (float): Maximum alpha.
            * pheromone_quantity (float): Quantity of pheromone.
            **dparams (Dict[str, Any]): Additional arguments.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
                1. New population
                2. New population fitness/function values
                3. Additional arguments:
                    * best_solution (float): Initial best solution.
                    * best_vector (numpy.ndarray): Initial best vector.
                    * direction_of_movement (int): Direction of ants movement.
                    * x_previous (float): Previous best solution.
                    * x_previous_vector (numpy.ndarray): Previous best vector.
                    * alphaMin (float): Minimum alpha.
                    * alphaMax (float): Maximum alpha.
                    * pheromone_quantity (float): Quantity of pheromone.
        """
        for i, w in enumerate(pop):
            dx = self.uniform(alphaMin, alphaMax, task.D)
            if direction_of_movement == 1:
                pop[i] = x_previous_vector + dx
            elif direction_of_movement == -1:
                pop[i] = x_previous_vector - dx
        for i, w in enumerate(pop):
            pop[i] = task.repair(pop[i])
            fpop[i] = task.eval(pop[i])
        x_best_vector, x_best_solution = self.getBest(pop, fpop)
        if x_best_solution <= x_previous:
            best_solution, best_vector = x_best_solution, x_best_vector
        else:
            best_solution, best_vector = x_previous, x_previous_vector
        pheromone_quantity = pheromone_quantity * self.evaporation
        pheromone_quantity = pheromone_quantity + (0.01 * x_best_solution)
        x_previous, x_previous_vector = x_best_solution, x_best_vector
        alphaMin, alphaMax = self.jumpLength * alphaMin, self.jumpLength * alphaMax
        return pop, fpop, {'best_solution': best_solution, 'best_vector': best_vector, 'direction_of_movement': direction_of_movement, 'x_previous': x_previous, 'x_previous_vector': x_previous_vector, 'alphaMin': alphaMin, 'alphaMax': alphaMax, 'pheromone_quantity': pheromone_quantity}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3

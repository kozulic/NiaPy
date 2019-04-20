# encoding=utf8
# pylint: disable=too-many-function-args
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.basic import AntColonyOptimization

class ACOTestCase(AlgorithmTestCase):
    def test_parameter_type(self):
        d = AntColonyOptimization.typeParameters()
        self.assertTrue(d['alphaMin'](-0.1))
        self.assertTrue(d['alphaMax'](11))
        self.assertTrue(d['jumpLength'](0.7))
        self.assertFalse(d['jumpLength'](0))
        self.assertTrue(d['pheromone'](0.3))
        self.assertFalse(d['pheromone'](-0.3))
        self.assertTrue(d['evaporation'](0.1))
        self.assertFalse(d['evaporation'](3))
        self.assertTrue(d['NP'](10))
        self.assertFalse(d['NP'](-10))
        self.assertFalse(d['NP'](0))

    def test_custom_works_fine(self):
        aco_custom = AntColonyOptimization(NP=40, alphaMin=-1.0, alphaMax=1.0, jumpLength=0.9, pheromone=0, evaporation=0.1, seed=self.seed)
        aco_customc = AntColonyOptimization(NP=40, alphaMin=-1.0, alphaMax=1.0, jumpLength=0.9, pheromone=0, evaporation=0.1, seed=self.seed)
        AlgorithmTestCase.algorithm_run_test(self, aco_custom, aco_customc, MyBenchmark())

    def test_griewank_works_fine(self):
        aco_griewank = AntColonyOptimization(NP=40, alphaMin=-1.0, alphaMax=1.0, jumpLength=0.9, pheromone=0, evaporation=0.1, seed=self.seed)
        aco_griewankc = AntColonyOptimization(NP=40, alphaMin=-1.0, alphaMax=1.0, jumpLength=0.9, pheromone=0, evaporation=0.1, seed=self.seed)
        AlgorithmTestCase.algorithm_run_test(self, aco_griewank, aco_griewankc)
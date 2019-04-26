# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

from NiaPy.algorithms.basic import AntColonyOptimization
from NiaPy.util import StoppingTask, OptimizationType
from NiaPy.benchmarks import Sphere
from NiaPy.benchmarks import Ackley
from NiaPy.benchmarks import Griewank

# we will run Ant Colony Optimization for 5 independent runs
for i in range(5):
    task = StoppingTask(D=10, nGEN=1000, optType=OptimizationType.MINIMIZATION, benchmark=Griewank())
    algo = AntColonyOptimization(NP=40)
    best = algo.run(task=task)
    print('%s -> %s' % (best[0], best[1]))

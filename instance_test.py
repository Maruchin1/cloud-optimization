import numpy as np
from independend_optimization import IndependentOptimization
from evolutionary_optimization import EvolutionaryOptimization
from solver import Solver
import time
from multiprocessing import Process


def execute(instance):
    print('Executing test')
    print()

    optimization_test = InstanceTest(instance, 'Optimization Test')
    optimization_test.execute_optimization()
    optimization_test.print_results()

    solver_test = InstanceTest(instance, 'Solver Test')
    solver_test.execute_solver()
    solver_test.print_results()

    return optimization_test.score, optimization_test.time, solver_test.score, solver_test.time


class InstanceTest:

    def __init__(self, instance, test_name):
        self.instance = instance
        self.test_name = test_name
        self.timeout_minutes = 5
        self.solution = None
        self.time = None
        self.score = None

    def print_results(self):
        print(self.test_name)
        print('Time:', self.time)
        print('Score', self.score)
        print('Solution:')
        print(self.solution)
        print()

    def execute_optimization(self):
        self._execute_with_timeout(self._execute_optimization())

    def execute_solver(self):
        self._execute_with_timeout(self._execute_solver())

    def _execute_with_timeout(self, fun_to_execute):
        p = Process(target=fun_to_execute, name=self.test_name)
        p.start()
        p.join(timeout=self.timeout_minutes * 60 * 1000)
        p.terminate()

    def _execute_optimization(self):
        t_start = time.time()
        independent = IndependentOptimization(self.instance)
        independent_solution = independent.execute()
        evolutionary = EvolutionaryOptimization(independent_solution, self.instance)
        self.solution = evolutionary.execute()
        self.time = time.time() - t_start
        self._calculate_score()

    def _execute_solver(self):
        t_start = time.time()
        solver = Solver(self.instance)
        self.solution = solver.execute()
        self.time = time.time() - t_start
        self._calculate_score()

    def _calculate_score(self):
        if self.solution is not None:
            max_times = np.max((self.solution * self.instance.p) * np.sum(self.solution, axis=0), axis=1)
            sums_of_expenses = np.sum(self.solution * (self.instance.t_peak * self.instance.p), axis=1)
            self.score = np.sum(max_times) + np.sum(sums_of_expenses)

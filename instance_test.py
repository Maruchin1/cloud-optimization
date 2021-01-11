import numpy as np
from independend_optimization import IndependentOptimization
from evolutionary_optimization import EvolutionaryOptimization
from solver import Solver
import time
from multiprocessing import Process, Value

# Timeout in seconds
timeout = 5 * 60

def execute(instance):
    print('Executing test')
    print()

    optimization_score = Value('f', -1.0)
    optimization_time = Value('f', -1.0)
    solver_score = Value('f', -1.0)
    solver_time = Value('f', -1.0)

    print('Testing optimization')
    optimization_process = Process(target=run_optimization, args=(instance, optimization_score, optimization_time))
    optimization_process.start()
    optimization_process.join(timeout=timeout)
    optimization_process.terminate()
    print('Score:', optimization_score.value)
    print('Time:', optimization_time.value)
    print()

    print('Testing solver')
    solver_process = Process(target=run_solver, args=(instance, solver_score, solver_time))
    solver_process.start()
    solver_process.join(timeout=timeout)
    solver_process.terminate()
    print('Score:', solver_score.value)
    print('Time:', solver_time.value)
    print()

    return optimization_score.value, optimization_time.value, solver_score.value, solver_time.value


def run_optimization(i, s, t):
    optimization_test = InstanceTest(i, 'Optimization Test')
    optimization_test.execute_optimization()
    s.value = optimization_test.score
    t.value = optimization_test.time


def run_solver(i, s, t):
    solver_test = InstanceTest(i, 'Solver Test')
    solver_test.execute_solver()
    s.value = solver_test.score
    t.value = solver_test.time


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

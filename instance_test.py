import numpy as np
from independend_optimization import IndependentOptimization
from evolutionary_optimization import EvolutionaryOptimization
from solver import Solver


def execute(instance):
    optimization_solution = _execute_optimization(instance)
    optimization_score = _calculate_score(instance, optimization_solution)
    print('Optimization solution')
    print(optimization_solution)
    print('Score:', optimization_score)
    print()

    solver_solution = _execute_solver(instance)
    solver_score = _calculate_score(instance, solver_solution)
    print('Solver solution')
    print(solver_solution)
    print('Score:', solver_score)
    print()


def _execute_optimization(instance):
    independent = IndependentOptimization(instance)
    independent_solution = independent.execute()

    evolutionary = EvolutionaryOptimization(independent_solution, instance)
    evolutionary_solution = evolutionary.execute()

    return evolutionary_solution


def _execute_solver(instance):
    solver = Solver(instance)
    solver_solution = solver.execute()
    return solver_solution


def _calculate_score(instance, solution):
    max_times = np.max((solution * instance.p) * np.sum(solution, axis=0), axis=1)
    sums_of_expenses = np.sum(solution * (instance.t_peak * instance.p), axis=1)
    return np.sum(max_times) + np.sum(sums_of_expenses)

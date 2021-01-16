import pandas as pd
import numpy as np
from instance_generator import ProblemInstance
from instance_test import execute_test

# Constants
p = [1.0, 1.2, 1.4, 1.6, 1.7, 1.8, 2.0]
w_t = 0.5
w_e = 0.5

# Generator
s = 4
t_peak_bounds = (2, 6)


def generate_instances(subtasks_bounds, num_of_instances):
    instances_arr = []
    for i in range(num_of_instances):
        random_t_peak = np.random.uniform(low=t_peak_bounds[0], high=t_peak_bounds[1], size=(s, len(p)))
        for row in random_t_peak:  # sort descending each row
            row[::-1].sort()
        random_subtasks = np.random.randint(low=subtasks_bounds[0], high=subtasks_bounds[1], size=s)
        T = random_subtasks * 15
        M = T + 20
        instance = ProblemInstance(p, random_t_peak, random_subtasks, T, M, w_t, w_e)
        instances_arr.append(instance)
    return instances_arr


def save_results(name, results_arr):
    frame = pd.DataFrame({
        'optimization_score': map(lambda x: x[0], results_arr),
        'solver_score': map(lambda x: x[1], results_arr),
        'optimization_time': map(lambda x: x[2], results_arr),
        'solver_time': map(lambda x: x[3], results_arr)
    })
    frame.to_excel(name + '_results.xlsx')


def run_experiment(name, subtasks_bounds, num_of_instances):
    print('-- Starting experiment:', name)
    instances_arr = generate_instances(subtasks_bounds, num_of_instances)
    print('-- Instances generated')

    print('-- Starting test loop')
    results_arr = []
    for num, instance in enumerate(instances_arr, start=1):
        print('-- Testing instance', name, num)
        result = execute_test(instance)
        results_arr.append(result)
    print('-- Testing finished')

    save_results(name, results_arr)
    print('-- Results saved')


if __name__ == '__main__':
    run_experiment('subtasks_1-3', (1, 3), 100)

import numpy as np
import json
import os
from instance_generator import ProblemInstance, InstanceGenerator
from instance_test import execute
import pandas as pd


def main():
    generate_small_instances()
    generate_medium_instances()
    df_results = pd.DataFrame()
    columns = ['file', 'optimization_score', 'solver_score', 'optimization_time', 'solver_time']
    for file in os.listdir('./instances'):
        instance = read_instance_from_file(file[:-5])
        (optimization_score, optimization_time, solver_score, solver_time) = execute(instance)
        df = pd.DataFrame([[file, optimization_score, solver_score, optimization_time, solver_time]], columns=columns)
        df_results = df_results.append(df, ignore_index=True)
    # save results to excel
    df_results.to_excel('results.xlsx')


def generate_small_instances():
    ig = InstanceGenerator()
    ig.instance_name = 'small'

    ig.m = 5
    ig.s = 3

    ig.T_low = 20
    ig.T_high = 40

    ig.M_low = 50
    ig.M_high = 70

    ig.save_to_file()


def generate_medium_instances():
    ig = InstanceGenerator()
    ig.instance_name = 'medium'

    ig.m = 10
    ig.s = 6

    ig.T_low = 40
    ig.T_high = 80

    ig.M_low = 100
    ig.M_high = 140

    ig.save_to_file()


def generate_big_instances():
    ig = InstanceGenerator()
    ig.instance_name = 'big'

    ig.m = 15
    ig.s = 9

    ig.T_low = 60
    ig.T_high = 120

    ig.M_low = 150
    ig.M_high = 210

    ig.save_to_file()


def read_instance_from_file(file_name):
    with open(os.path.join("instances", file_name + ".json"), 'r') as f:
        instance = json.loads(json.load(f))
    return ProblemInstance(
        np.asarray(instance['p']),
        np.asarray(instance['t_peak']),
        np.asarray(instance['tasks_subtasks']),
        np.asarray(instance['T']),
        np.asarray(instance['M']),
        instance['w_t'],
        instance['w_e']
    )


if __name__ == "__main__":
    main()

import pandas as pd
from instance_generator import InstanceGenerator
from instance_test import execute


test_name = 'score_medium'
num_of_machines = 10
num_of_tasks = 6
num_of_instances = 10


def run_experiment():
    print('Starting experiment:', test_name)
    generator = InstanceGenerator()
    generator.test_name = test_name
    generator.m = num_of_machines
    generator.s = num_of_tasks
    generator.num_of_instances = num_of_instances
    instances_arr = generator.generate_instances()
    print('Instances generated')

    print('Starting Test Loop')
    results_storage = ResultsStorage()
    for num, instance in enumerate(instances_arr, start=1):
        print('Testing instance', num)
        (optimization_score, _, solver_score, _) = execute(instance)
        results_storage.add_result(optimization_score, solver_score)
    print('Testing finished')

    results_storage.save_to_excel()
    print('Results saved')


class ResultsStorage:

    def __init__(self):
        self.df_results = pd.DataFrame()

    def add_result(self, optimization_score, solver_score):
        df_single_result = pd.DataFrame(
            data=[[optimization_score, solver_score]],
            columns=['optimization_score', 'solver_score']
        )
        self.df_results = self.df_results.append(df_single_result, ignore_index=True)

    def save_to_excel(self):
        self.df_results.to_excel(test_name + '_results.xlsx')


if __name__ == '__main__':
    run_experiment()

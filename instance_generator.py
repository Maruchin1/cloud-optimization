import json
import numpy as np


# w_t - waga czasu
# w_e - waga kosztu
# m - liczba maszyn
# s - liczba tasków
# p_high - maksymalna wartość p (minimalna jest 1)
# T_low - minimalne ograniczenie czasowe
# T_high - maksymalne ograniczenie czasowe
# M_low - minimalne kosztów
# M_high- maksymalne ograniczenie kosztów


class InstanceGenerator:

    def __init__(self):
        self.test_name = 'test'
        self.m = 5
        self.s = 3
        self.p_low = 1
        self.p_high = 2
        self.t_peak_low = 2
        self.t_peak_high = 6
        self.w_t = 0.5
        self.w_e = 0.5
        self.num_of_instances = 10

    def generate_instances(self):
        instances_arr = []
        for i in range(self.num_of_instances):
            instance = self._generate_single_instance()
            instances_arr.append(instance)
        return instances_arr

    def _generate_single_instance(self):
        p = np.random.uniform(low=self.p_low, high=self.p_high, size=(self.m,))
        p = np.sort(p)

        t_peak = np.random.uniform(low=self.t_peak_low, high=self.t_peak_high, size=(self.s, self.m))
        for row in t_peak:  # sort descending each row
            row[::-1].sort()

        tasks_subtasks = np.random.randint(low=1, high=self.m, size=self.s)
        tasks_subtasks = np.sort(tasks_subtasks)

        T = tasks_subtasks * 15
        M = T * 2

        return ProblemInstance(p, t_peak, tasks_subtasks, T, M, self.w_t, self.w_e)


class ProblemInstance:
    def __init__(self, p, t_peak, tasks_subtasks, T, M, w_t, w_e):
        self.p = p
        self.t_peak = t_peak
        self.tasks_subtasks = tasks_subtasks
        self.T = T
        self.M = M
        self.w_t = w_t
        self.w_e = w_e


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

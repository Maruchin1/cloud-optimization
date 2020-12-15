import json
import numpy as np
import pathlib
import os


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
        self.m = 5
        self.s = 3
        self.w_t = 0.5
        self.w_e = 0.5
        self.p_high = 2
        self.t_peak_low = 2
        self.t_peak_high = 6
        self.T_low = 40
        self.T_high = 70
        self.M_low = 70
        self.M_high = 100
        self.num_of_problem_instances = 3
        self.instance_name = 'testowa'
        self.instances_dir = 'instances'

    def save_to_file(self):
        self._check_if_dir_created()
        for i in range(self.num_of_problem_instances):
            instance = self._generate_problem_instance()
            json_instance = json.dumps(instance.__dict__, cls=NumpyEncoder)
            with open(os.path.join("instances", self.instance_name + str(i) + ".json"), 'w') as f:
                json.dump(json_instance, f)

    def _check_if_dir_created(self):
        instances_dir = pathlib.Path(self.instances_dir)
        if not instances_dir.exists():
            os.mkdir(self.instances_dir)

    def _generate_problem_instance(self):
        p = np.random.uniform(low=1, high=self.p_high, size=(self.m,))
        p = np.sort(p)

        t_peak = np.random.uniform(low=self.t_peak_low, high=self.t_peak_high, size=(self.s, self.m))
        for row in t_peak:  # sort descending each row
            row[::-1].sort()

        tasks_subtasks = np.random.randint(low=1, high=self.m, size=self.s)
        tasks_subtasks = np.sort(tasks_subtasks)

        T = np.random.randint(low=self.T_low, high=self.T_high, size=self.s)
        T = np.sort(T)

        M = np.random.randint(low=self.M_low, high=self.M_high, size=self.s)
        M = np.sort(M)

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

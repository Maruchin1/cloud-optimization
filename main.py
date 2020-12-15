import numpy as np
from ProblemInstance import ProblemInstance
import json
import os
from instance_test import execute
import pandas as pd


# w_t - waga czasu
# w_e - waga kosztu
# m - liczba maszyn
# s - liczba tasków
# p_high - maksymalna wartość p (minimalna jest 1)
# T_low - minimalne ograniczenie czasowe
# T_high - maksymalne ograniczenie czasowe
# M_low - minimalne kosztów
# M_high- maksymalne ograniczenie kosztów

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def generateProblemInstance(w_t, w_e, m, s, p_high, t_peak_low, t_peak_high, T_low, T_high, M_low, M_high):
    p = np.random.uniform(low=1, high=p_high, size=(m,))
    p = np.sort(p)

    t_peak = np.random.uniform(low=t_peak_low, high=t_peak_high, size=(s, m))
    for row in t_peak:  # sort descending each row
        row[::-1].sort()

    tasks_subtasks = np.random.randint(low=1, high=m, size=s)
    tasks_subtasks = np.sort(tasks_subtasks)

    T = np.random.randint(low=T_low, high=T_high, size=s)
    T = np.sort(T)

    M = np.random.randint(low=M_low, high=M_high, size=s)
    M = np.sort(M)

    return ProblemInstance(p, t_peak, tasks_subtasks, T, M, w_t, w_e)


def saveToFile(w_t, w_e, m, s, p_high, t_peak_low, t_peak_high, T_low, T_high, M_low, M_high, num_of_problem_instances,
               instanceName):
    createDirectory("instances")

    for i in range(num_of_problem_instances):
        instance = generateProblemInstance(w_t, w_e, m, s, p_high, t_peak_low, t_peak_high, T_low, T_high, M_low,
                                           M_high)
        jsonInstance = json.dumps(instance.__dict__, cls=NumpyEncoder)

        with open(os.path.join("instances", instanceName + str(i) + ".json"), 'w') as f:
            json.dump(jsonInstance, f)


def createDirectory(path):
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)


def readInstanceFromFile(instanceName):
    with open(os.path.join("instances", instanceName + ".json"), 'r') as f:
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


def main():
    # Generate test instances
    saveToFile(0.5, 0.5, 5, 3, 2, 1, 40, 20, 40, 20, 40, 2, 'testowa')

    df_results = pd.DataFrame()
    columns = ['file', 'optimization_score', 'solver_score', 'optimization_time', 'solver_time']
    for file in os.listdir('./instances'):
        instance = readInstanceFromFile(file[:-5])
        test_instance = ProblemInstance(
            p=np.array([1, 1.2, 1.5, 1.8, 2]),
            t_peak=np.array([
                [6, 5, 4, 3.5, 3],
                [5, 4.2, 3.6, 3, 2.8],
                [4, 3.5, 3.2, 2.8, 2.4]
            ]),
            tasks_subtasks=np.array([2, 3, 4]),
            T=np.array([20, 30, 40]),
            M=np.array([50, 60, 70]),
            w_t=0.5,
            w_e=0.5
        )
        (optimization_score, optimization_time, solver_score, solver_time) = execute(instance)
        df = pd.DataFrame([[file, optimization_score, solver_score, optimization_time, solver_time]], columns=columns)
        df_results = df_results.append(df, ignore_index=True)
    # save results to excel
    df_results.to_excel('results.xlsx')


if __name__ == "__main__":
    main()

import cvxpy
import numpy as np


class IndependentOptimization:

    def __init__(self, instance):
        self.t_peak = instance.t_peak
        self.tasks_subtasks = instance.tasks_subtasks
        self.p = instance.p
        self.w_t = instance.w_t
        self.w_e = instance.w_e

    def execute(self):
        a = np.zeros((self.t_peak.shape[0], self.t_peak.shape[1]))
        for i in range(len(self.tasks_subtasks)):
            a[i] = self._optimize_single(i)
        return a

    def _optimize_single(self, i):
        times_vector = self.t_peak[i]
        num_of_subtasks = self.tasks_subtasks[i]

        a = cvxpy.Variable(len(self.p), boolean=True)

        t = times_vector @ a
        e = t * self.p

        max_time = cvxpy.max(t)
        sum_of_expense = cvxpy.sum(e)

        u = max_time * self.w_t + sum_of_expense * self.w_e

        constraint = cvxpy.sum(a) == num_of_subtasks

        problem = cvxpy.Problem(cvxpy.Minimize(u), [constraint])
        problem.solve(solver=cvxpy.GLPK_MI)

        return a.value

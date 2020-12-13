import numpy as np


class EvolutionaryOptimization:

    def __init__(self, a, instance):
        self.a = a
        self.t_peak = instance.t_peak
        self.p = instance.p
        self.w_t = instance.w_t
        self.w_e = instance.w_e
        self.T = instance.T
        self.M = instance.M
        self.t = self._calculate_T(a)
        self.e = self._calcualte_E(self.t)

    def execute(self):
        task_index = 0
        flag = True

        while flag:

            if task_index == 0:
                flag = False

            ms = self._get_index_order(task_index)
            for j in range(len(ms)):

                q = self._min_global(ms[j])  # task id

                if q != -1:
                    p = self._min_single(q, ms[j])  # resource id
                    _spelr_reloc(self.a[q], ms[j], p)
                    flag = True

            if task_index == len(self.a) - 1:
                if not flag:
                    return self.a
                else:
                    task_index = 0
            else:
                task_index = task_index + 1

        return self.a

    def _min_global(self, current_resource_index):
        to_return = -1

        mts = self.a[:, current_resource_index]
        nsts = []

        for taskIndex in range(len(mts)):

            reloc_resource_index = self._min_single(taskIndex, current_resource_index)

            if reloc_resource_index != -1:
                utility_task_index = self._compute_utility_for_row(taskIndex)
                _spelr_reloc(self.a[taskIndex], current_resource_index, reloc_resource_index)
                utility_task_index_new = self._compute_utility_for_row(taskIndex)
                _spelr_reloc(self.a[taskIndex], reloc_resource_index, current_resource_index)

                if (utility_task_index - utility_task_index_new) < 0:
                    nsts.append((taskIndex, reloc_resource_index))

        if len(nsts) == 0:
            return -1

        else:
            utility = self._calculate_global_utility()
            min = utility
            for taskIndex, reloc_resource_index in nsts:
                _spelr_reloc(self.a[taskIndex], current_resource_index, reloc_resource_index)
                utility_new = self._calculate_global_utility()
                _spelr_reloc(self.a[taskIndex], reloc_resource_index, current_resource_index)

                diff = utility - utility_new

                if diff < min and self._fulfills_constraints():
                    min = diff
                    to_return = taskIndex

        return to_return

    def _min_single(self, task_index, current_resource_index):
        utility_task_index = self._compute_utility_for_row(task_index)
        min = 0
        reloc_resource_index = -1

        for resourceIndex in range(len(self.a.T)):

            if resourceIndex != current_resource_index:

                _spelr_reloc(self.a[task_index], current_resource_index, resourceIndex)
                utility_task_index_new = self._compute_utility_for_row(task_index)
                _spelr_reloc(self.a[task_index], resourceIndex, current_resource_index)

                diff = utility_task_index - utility_task_index_new

                if (diff < 0) and (diff < min):
                    reloc_resource_index = resourceIndex
                    min = diff

        return reloc_resource_index

    def _fulfills_constraints(self):
        t = self._calculate_T(self.a)
        e = self._calcualte_E(t)

        max_time = np.max(t, axis=1)
        sum_of_expense = np.sum(e, axis=1)

        for i in range(len(max_time)):
            if self.T[i] - max_time[i] < 0:
                return False
            if self.M[i] - sum_of_expense[i] < 0:
                return False
        return True

    def _compute_utility_for_row(self, task_index):
        t = self._calculate_T(self.a)
        e = self._calcualte_E(t)

        max_time = np.max(t[task_index])
        sum_of_expense = np.sum(e[task_index])

        max_time = max_time * self.w_t
        sum_of_expense = sum_of_expense * self.w_e

        return 1 / (max_time + sum_of_expense)

    def _calculate_global_utility(self):
        t = self._calculate_T(self.a)
        e = self._calcualte_E(t)

        max_time = np.max(t)
        sum_of_expense = np.sum(e)

        max_time = max_time * self.w_t
        sum_of_expense = sum_of_expense * self.w_e

        utility_vector = 1 / (max_time + sum_of_expense)

        return np.sum(utility_vector)

    def _get_index_order(self, task_index):
        return np.flip(np.argsort(self.t[task_index]))

    def _calculate_T(self, a):
        t = np.array(self.t_peak) * a
        multipler = a.sum(axis=0)
        return t * multipler

    def _calcualte_E(self, t):
        e = np.array(t) * self.p
        divider = self.a.sum(axis=0)

        for i in range(len(divider)):
            if divider[i] == 0:
                divider[i] = 1

        return e / divider


def _spelr_reloc(vector, p, j):
    temp = vector[p]
    vector[p] = vector[j]
    vector[j] = temp

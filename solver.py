import cvxpy


class Solver:

    def __init__(self, instance):
        self.num_of_tasks = len(instance.tasks_subtasks)
        self.num_of_machines = len(instance.p)
        self.t_peak = instance.t_peak
        self.p = instance.p
        self.w_t = instance.w_t
        self.w_e = instance.w_e
        self.tasks_subtasks = instance.tasks_subtasks
        self.T = instance.T
        self.M = instance.M

    def execute(self):
        # Zmienna decyzyjna
        a = cvxpy.Variable((self.num_of_tasks, self.num_of_machines), boolean=True)

        # Czasy wykonywania zadań
        sum = cvxpy.sum(a, axis=0, keepdims=True)
        sum_matrix = cvxpy.hstack([sum for i in range(self.num_of_tasks)])
        sum_matrix = cvxpy.reshape(sum_matrix, (self.num_of_tasks, self.num_of_machines))
        t = cvxpy.multiply(sum_matrix, self.t_peak)
        max_time = cvxpy.max(t, axis=1)

        # Koszty użycia maszyn
        e = cvxpy.multiply(a, self.t_peak * self.p)
        sum_of_expense = cvxpy.sum(e, axis=1)

        # Funkcja celu
        u = cvxpy.sum(max_time * self.w_t + sum_of_expense * self.w_e, axis=0, keepdims=True)

        # Ograniczenia
        subtasks_const = cvxpy.sum(a, axis=1).__eq__(self.tasks_subtasks)
        time_const = max_time.__le__(self.T)
        cost_const = sum_of_expense.__le__(self.M)

        # Rozwiązanie problemu
        problem = cvxpy.Problem(cvxpy.Minimize(u), [subtasks_const, time_const, cost_const])
        problem.solve(solver=cvxpy.GLPK_MI)

        return a.value

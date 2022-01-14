import math
import ml_lib.utils.linalg as linalg
import unittest


class LinearAlgebraTest(unittest.TestCase):

    def test_solve_sys_equations(self):
        a = [[1, 1, 1], [0, 2, 5], [2, 5, -1]]
        b = [6, -4, 27]
        solution = linalg.solve_sys_equations(a, b)
        print("Solution: %s" % str(solution))
        check_sol = [5., 3., -2.]
        for x, y in zip(check_sol, solution):
            self.assertTrue(math.isclose(x, y))

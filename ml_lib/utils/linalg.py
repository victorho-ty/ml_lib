import numpy as np

"""
A = [[a, b],
     [c, d]]
Inverse formula:
    inv(A) = (1 / (ad−bc)) *　[[d, —b],
                            　 [—c, a]]
"""


def solve_sys_equations(a, b):
    """
    A X = B
    X = inv(A) * B
    """
    A = np.array(a)
    B = np.array(b)
    A_inv = np.linalg.inv(A)
    solution = np.matmul(A_inv, B)
    return solution


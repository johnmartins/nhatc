import numpy as np
from scipy.optimize import minimize


def obj_f(x, args):
    print("OBJ")
    x1 = x[0]
    x2 = x[1]
    return x1**2 + x1*x2


def eq_const(x, *args):
    print("EQ")
    x1 = x[0]
    x2 = x[1]
    return x1**3 + x1 * x2 - 100


def ineq_const1(x, *args):
    print("INEQ")
    x1 = x[0]
    x2 = x[1]
    return 50 - np.abs(x1) - np.abs(x2)


bounds_x1 = [-100, 100]
bounds_x2 = [-100, 100]
bounds = [bounds_x1, bounds_x2]

arguments=[1,2,3]

constraints = [
    {'type': 'eq', 'fun': eq_const, 'args': arguments},
    {'type': 'ineq', 'fun': ineq_const1, 'args': arguments},
]

x0 = np.array([10, 10])

res = minimize(obj_f, x0,
               args=arguments,
               method=None,
               bounds=bounds,
               constraints=constraints)

print(res)

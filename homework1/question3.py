import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.Line_Search import wolfe_search
from src.Descent_Method import steepest_descent, pure_newton_method

def Rosenbrock_function(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def Rosenbrock_gradient(x):
    dfdx0 = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    dfdx1 = 200 * (x[1] - x[0]**2)
    return np.array([dfdx0, dfdx1])

def Rosenbrock_hessian(x):
    d2fdx0dx0 = -400 * (x[1] - x[0]**2) + 800 * x[0]**2 + 2
    d2fdx0dx1 = -400 * x[0]
    d2fdx1dx1 = 200
    return np.array([[d2fdx0dx0, d2fdx0dx1], [d2fdx0dx1, d2fdx1dx1]])

def Beale_function(x):
    return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2

def Beale_gradient(x):
    dfdx0 = 2 * (1.5 - x[0] + x[0]*x[1]) * (-1 + x[1]) + 2 * (2.25 - x[0] + x[0]*x[1]**2) * (-1 + x[1]**2) + 2 * (2.625 - x[0] + x[0]*x[1]**3) * (-1 + x[1]**3)
    dfdx1 = 2 * (1.5 - x[0] + x[0]*x[1]) * (x[0]) + 4 * (2.25 - x[0] + x[0]*x[1]**2) * (x[0]*x[1]) + 6 * (2.625 - x[0] + x[0]*x[1]**3) * (x[0]*x[1]**2)
    return np.array([dfdx0, dfdx1])

def Beale_hessian(x):
    d2fdx0dx0 = 2 * (-1 + x[1])**2 + 2 * (-1 + x[1]**2)**2 + 2 * (-1 + x[1]**3)**2
    d2fdx0dx1 = 2 * (-1 + x[1]) * x[0] + 2 * (1.5 - x[0] + x[0]*x[1]) * 1 + 2 * (-1 + x[1]**2) * (2 * x[0] * x[1]) + 2 * (2.25 - x[0] + x[0]*x[1]**2) * (2 * x[1]) + 2 * (-1 + x[1]**3) * (3 * x[0] * x[1]**2) + 2 * (2.625 - x[0] + x[0]*x[1]**3) * (3 * x[1]**2)
    d2fdx1dx1 = 2 * x[0] * x[0] + 2 * (2 * x[0] * x[1]) * (2 * x[0] * x[1]) + 2 * (2.25 - x[0] + x[0]*x[1]**2) * (2 * x[0]) + 2 * (3 * x[0] * x[1]**2) * (3 * x[0] * x[1]**2) + 2 * (2.625 - x[0] + x[0]*x[1]**3) * (6 * x[0] * x[1])
    return np.array([[d2fdx0dx0, d2fdx0dx1], [d2fdx0dx1, d2fdx1dx1]])

if __name__ == "__main__":
    x_0 = np.array([-1.2, 1.0])
    tol = 1e-5
    max_iter = 10000
    print("="*50)
    print("Rosenbrock Function")
    print("="*50)
    print("最速下降法:")
    x_min, iterations = steepest_descent(Rosenbrock_function, Rosenbrock_gradient, x_0, tol=tol, max_iter=max_iter)
    print(f"迭代次数: {iterations}")
    print(f"数值解: {x_min}")
    print(f"最终梯度范数: {np.linalg.norm(Rosenbrock_gradient(x_min)):.2e}")
    print(f"目标函数值: {Rosenbrock_function(x_min):.6f}")

    print("\n")
    print("纯牛顿法:")
    x_min, iterations = pure_newton_method(Rosenbrock_function, Rosenbrock_gradient, Rosenbrock_hessian, x_0, tol=tol, max_iter=max_iter)
    print(f"迭代次数: {iterations}")
    print(f"数值解: {x_min}")
    print(f"最终梯度范数: {np.linalg.norm(Rosenbrock_gradient(x_min)):.2e}")
    print(f"目标函数值: {Rosenbrock_function(x_min):.6f}")

    print("="*50)
    print("Beale Function")
    print("="*50)
    print("最速下降法:")
    x_min, iterations = steepest_descent(Beale_function, Beale_gradient, x_0, tol=tol, max_iter=max_iter)
    print(f"迭代次数: {iterations}")
    print(f"数值解: {x_min}")
    print(f"最终梯度范数: {np.linalg.norm(Beale_gradient(x_min)):.2e}")
    print(f"目标函数值: {Beale_function(x_min):.6f}")

    print("\n")
    print("纯牛顿法:")
    x_min, iterations = pure_newton_method(Beale_function, Beale_gradient, Beale_hessian, x_0, tol=tol, max_iter=max_iter)
    print(f"迭代次数: {iterations}")
    print(f"数值解: {x_min}")
    print(f"最终梯度范数: {np.linalg.norm(Beale_gradient(x_min)):.2e}")
    print(f"目标函数值: {Beale_function(x_min):.6f}")

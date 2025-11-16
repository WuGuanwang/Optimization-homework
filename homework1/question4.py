import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.Line_Search import wolfe_search
from src.Descent_Method import pure_newton_method,newton_method_with_linesearch

A=np.array([[5,1,0,0.5],[1,4,0.5,0],[0,0.5,3,0],[0.5,0,0,2]])
rho = 10000

def A_function(x):
    return 0.5 * np.dot(x,x) + 0.25 * (np.dot(x, A @ x) ** 2)

def A_gradient(x):
    Ax = A @ x
    return x + (np.dot(x, Ax)) * Ax

def A_hessian(x):
    Ax = A @ x
    outer_Ax = np.outer(Ax, Ax)
    return np.eye(len(x)) + np.dot(x, Ax) * A + 2 * outer_Ax

def A_rho_function(x):
    return 0.5 * np.dot(x,x) + 0.25 * rho * (np.dot(x, A @ x) ** 2)

def A_rho_gradient(x):
    Ax = A @ x
    return x + (np.dot(x, Ax)) * Ax * rho

def A_rho_hessian(x):
    Ax = A @ x
    outer_Ax = np.outer(Ax, Ax)
    return np.eye(len(x)) + np.dot(x, Ax) * A * rho + 2 * outer_Ax * rho

if __name__ == "__main__":
    tol=1e-5
    max_iter=1000
    x_0 = np.array([np.cos(7/18), np.sin(7/18), np.cos(7/18), np.sin(7/18)])
    print("="*50)
    print("when x_0 =[cos(7/18), sin(7/18), cos(7/18), sin(7/18)]")
    print("="*50)
    print("when rho = 1")

    print("纯牛顿法:")
    x_min, iterations = pure_newton_method(A_function, A_gradient, A_hessian, x_0, tol=tol, max_iter=max_iter)
    print(f"迭代次数: {iterations}")
    print(f"数值解: {x_min}")
    print(f"最终梯度范数: {np.linalg.norm(A_gradient(x_min)):.2e}")
    print(f"目标函数值: {A_function(x_min):.6f}")

    print("带线搜索的牛顿法:")
    x_min, iterations = newton_method_with_linesearch(A_function, A_gradient, A_hessian, x_0, tol=tol, max_iter=max_iter)
    print(f"迭代次数: {iterations}")
    print(f"数值解: {x_min}")
    print(f"最终梯度范数: {np.linalg.norm(A_gradient(x_min)):.2e}")
    print(f"目标函数值: {A_function(x_min):.6f}")

    print("-" * 50 )
    print("\nwhen rho = 10000")
    print("纯牛顿法:")
    x_min, iterations = pure_newton_method(A_rho_function, A_rho_gradient, A_rho_hessian, x_0, tol=tol, max_iter=max_iter)
    print(f"迭代次数: {iterations}")
    print(f"数值解: {x_min}")
    print(f"最终梯度范数: {np.linalg.norm(A_rho_gradient(x_min)):.2e}")
    print(f"目标函数值: {A_rho_function(x_min):.6f}")

    print("带线搜索的牛顿法:")
    x_min, iterations = newton_method_with_linesearch(A_rho_function, A_rho_gradient, A_rho_hessian, x_0, tol=tol, max_iter=max_iter)
    print(f"迭代次数: {iterations}")
    print(f"数值解: {x_min}")
    print(f"最终梯度范数: {np.linalg.norm(A_rho_gradient(x_min)):.2e}")
    print(f"目标函数值: {A_rho_function(x_min):.6f}")

    x_0 = np.array([np.cos(5/18), np.sin(5/18), np.cos(5/18), np.sin(5/18)])
    print("\n"+"="*50)
    print("\nwhen x_0 =[cos(5/18), sin(5/18), cos(5/18), sin(5/18)]")
    print("="*50)
    print("when rho = 1")
    print("纯牛顿法:")
    x_min, iterations = pure_newton_method(A_function, A_gradient, A_hessian, x_0, tol=tol, max_iter=max_iter)
    print(f"迭代次数: {iterations}")
    print(f"数值解: {x_min}")
    print(f"最终梯度范数: {np.linalg.norm(A_gradient(x_min)):.2e}")
    print(f"目标函数值: {A_function(x_min):.6f}")

    print("带线搜索的牛顿法:")
    x_min, iterations = newton_method_with_linesearch(A_function, A_gradient, A_hessian, x_0, tol=tol, max_iter=max_iter)
    print(f"迭代次数: {iterations}")
    print(f"数值解: {x_min}")
    print(f"最终梯度范数: {np.linalg.norm(A_gradient(x_min)):.2e}")
    print(f"目标函数值: {A_function(x_min):.6f}")
    print("-" * 50 )
    print("\nwhen rho = 10000")
    print("纯牛顿法:")
    x_min, iterations = pure_newton_method(A_rho_function, A_rho_gradient, A_rho_hessian, x_0, tol=tol, max_iter=max_iter)
    print(f"迭代次数: {iterations}")
    print(f"数值解: {x_min}")
    print(f"最终梯度范数: {np.linalg.norm(A_rho_gradient(x_min)):.2e}")
    print(f"目标函数值: {A_rho_function(x_min):.6f}")

    print("带线搜索的牛顿法:")
    x_min, iterations = newton_method_with_linesearch(A_rho_function, A_rho_gradient, A_rho_hessian, x_0, tol=tol, max_iter=max_iter)
    print(f"迭代次数: {iterations}")
    print(f"数值解: {x_min}")
    print(f"最终梯度范数: {np.linalg.norm(A_rho_gradient(x_min)):.2e}")
    print(f"目标函数值: {A_rho_function(x_min):.6f}")
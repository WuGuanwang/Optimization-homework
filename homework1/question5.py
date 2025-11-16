import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.Line_Search import wolfe_search
from src.Descent_Method import steepest_descent,Barzilar_Borwein_method
from scipy.sparse import random as sparse_random

def l_sigma_function(x, delta):
    return np.where(np.abs(x) < delta, x**2/(2*delta), np.abs(x) - delta/2)

def l_sigma_gradient(x, delta):
    return np.where(np.abs(x) < delta, x/delta, np.sign(x))

def L_sigma_function(x, delta):
    return np.sum(l_sigma_function(x, delta))

def L_sigma_gradient(x, delta):
    return l_sigma_gradient(x, delta)

def LASSO_function(x,A,b,mu,delta):
    Ax_b = A @ x - b
    return 0.5 * np.dot(Ax_b, Ax_b) + mu * L_sigma_function(x,delta)

def LASSO_gradient(x,A,b,mu,delta):
    Ax_b = A @ x - b
    grad = A.T @ Ax_b
    grad += mu * L_sigma_gradient(x,delta)
    return grad

if __name__ == "__main__":
    np.random.seed(0)
    m, n = 512, 1024
    A = np.random.randn(m, n)
    x_true = sparse_random(n, 1, density=0.1, random_state=0).toarray().flatten()
    b = A @ x_true 
    x0 = np.zeros(n)
    tol = 1e-3
    max_iter = 50000
    print(f"问题维度: m={m}, n={n}")
    print(f"真实解的稀疏度: {np.sum(x_true != 0)}/{n}")
    

    for mu in [1e-2, 1e-3]:
        delta = 1e-3 * mu
        print(f"\n{'='*50}")
        print(f"mu = {mu}, delta = {delta}")
        print(f"{'='*50}")

        func = lambda x: LASSO_function(x, A, b, mu, delta)
        grad = lambda x: LASSO_gradient(x, A, b, mu, delta)

        print("最速下降法:")
        x_opt_sd, iterations_sd = steepest_descent(func, grad, x0, tol=tol, max_iter=max_iter)
        print(f"迭代次数: {iterations_sd}")
        print(f"最终梯度范数: {np.linalg.norm(grad(x_opt_sd)):.2e}")
        print(f"目标函数值: {func(x_opt_sd):.6f}")
        print(f"解的稀疏度: {np.sum(np.abs(x_opt_sd) > 1e-4)}/{n}")
        print(f"与真实解的误差: {np.linalg.norm(x_opt_sd - x_true):.6f}")
        
        print("\nBarzilai-Borwein 方法:")
        x_opt_bb, iterations_bb = Barzilar_Borwein_method(func, grad, x0, tol=tol, max_iter=max_iter)
        print(f"迭代次数: {iterations_bb}")
        print(f"最终梯度范数: {np.linalg.norm(grad(x_opt_bb)):.2e}")
        print(f"目标函数值: {func(x_opt_bb):.6f}")
        print(f"解的稀疏度: {np.sum(np.abs(x_opt_bb) > 1e-4)}/{n}")
        print(f"与真实解的误差: {np.linalg.norm(x_opt_bb - x_true):.6f}")
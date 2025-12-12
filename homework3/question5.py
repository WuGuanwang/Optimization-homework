import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.Descent_Method import newton_method_with_explicit_linesearch
import numpy as np
from src.draw import draw_residual_convergence,draw_gradient_convergence,translate


def f(x):
    return 0.5 * (1 - 2*x[1] + 2*x[1]**2  - 2*x[1]*x[0]**2 + x[0]**4)

def grad_f(x):
    dfdx0 = -2*x[0]*x[1] + 2*x[0]**3
    dfdx1 = -1 + 2*x[1] - x[0]**2
    return np.array([dfdx0, dfdx1])

def hess_f(x):
    d2fdx00 = -2*x[1] + 6*x[0]**2
    d2fdx01 = -2*x[0]
    d2fdx11 = 2
    return np.array([[d2fdx00, d2fdx01],
                     [d2fdx01, d2fdx11]])

if __name__ == "__main__":
    x_star = np.array([1.0,1.0])
    x_1 = np.array([3.0,2.0])   
    tol = 1e-6
    max_iter = 1000
    print("用Newton方法处理题5中的函数...")
    x,num_iter = newton_method_with_explicit_linesearch(f, grad_f, hess_f, x_1, tol=tol, max_iter=max_iter, save_full_data=True,
                filename="question5_newton_results.txt")
    print(f"最终迭代次数为： {num_iter}")
    print(f"最终解为： {x}")
    draw_gradient_convergence(filename="question5_newton_results.txt",save_plot=True,
                              plot_filename="question5_newton_gradient_convergence.png")
    translate(x_star=x_star,input_filename="question5_newton_results.txt",
              output_filename="question5_newton_residual_results.txt"
              )
    draw_residual_convergence(filename="question5_newton_residual_results.txt",save_plot=True,
                              plot_filename="question5_newton_residual_convergence.png")
    print("="*50)


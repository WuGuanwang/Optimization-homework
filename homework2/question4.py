import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.Descent_Method import BFGS_method
import numpy as np
from src.draw import draw_residual_convergence,draw_gradient_convergence,translate

def Rosenbrock_function(n,x):
    sum=0
    for i in range(n-1):
        sum += 100.0 * (x[i+1] - x[i]**2.0)**2.0 + (1 - x[i])**2.0
    return sum

def Rosenbrock_gradient(n,x):
    grad = np.zeros(n)
    for i in range(n-1):
        grad[i] += -400 * x[i] * (x[i+1] - x[i]**2) - 2 * (1 - x[i])
        grad[i+1] += 200 * (x[i+1] - x[i]**2)
    return grad


def Powell_singular_function(x):
    f1 = x[0] + 10 * x[1]
    f2 = (x[2] - x[3])
    f3 = (x[1] - 2 * x[2])**2
    f4 = (x[0] - x[3])**2
    return f1**2 + 5*f2**2 + f3**2 + 10*f4**2

def Powell_singular_gradient(x):
    grad = np.zeros(4)
    f1 = x[0] + 10 * x[1]
    f2 = (x[2] - x[3])
    f3 = (x[1] - 2 * x[2])**2
    f4 = (x[0] - x[3])**2

    grad[0] = 2*f1 + 40*f4*(x[0] - x[3])
    grad[1] = 20*f1 + 4*f3*(x[1] - 2*x[2])
    grad[2] = 10*f2 - 8*f3*(x[1] - 2*x[2])
    grad[3] = -10*f2 - 40*f4*(x[0] - x[3])
    return grad

if __name__ == "__main__":
    for n in [6,8,10]:
        print(f"用BFGS处理Rosenbrock函数,维度n={n}...")
        Rosenbrock_function_n=lambda x :Rosenbrock_function(n,x)
        Rosenbrock_gradient_n=lambda x :Rosenbrock_gradient(n,x)
        x0 = np.array([-1.2 if i % 2 == 1 else 1.0 for i in range(n)])
        x,num_iter = BFGS_method(Rosenbrock_function_n, Rosenbrock_gradient_n,x0,tol=1e-5, max_iter=1000,save_full_data=True,
                    filename=f"Rosenbrock_gradient_descent_n{n}_results.txt")
        print(f"最终迭代次数为： {num_iter}")
        print(f"最终解为： {x}")
        draw_gradient_convergence(filename=f"Rosenbrock_gradient_descent_n{n}_results.txt",save_plot=True,
                                  plot_filename=f"Rosenbrock_gradient_convergence_n{n}.png")
        translate(x_star=[1.0]*n,input_filename=f"Rosenbrock_gradient_descent_n{n}_results.txt",
                  output_filename=f"Rosenbrock_residual_descent_n{n}_results.txt"
                  )
        draw_residual_convergence(filename=f"Rosenbrock_residual_descent_n{n}_results.txt",save_plot=True,
                                  plot_filename=f"Rosenbrock_residual_convergence_n{n}.png")
        print("="*50)

    print("用BFGS处理Powell奇异函数...")
    x0 = np.array([3.0, -1.0, 0.0, 1.0])
    x, num_iter = BFGS_method(Powell_singular_function, Powell_singular_gradient,x0,tol=1e-5, max_iter=1000,save_full_data=True,
                filename="Powell_singular_gradient_descent_results.txt")
    print(f"最终迭代次数为： {num_iter}")
    print(f"最终解为： {x}")
    draw_gradient_convergence(filename="Powell_singular_gradient_descent_results.txt",save_plot=True,
                              plot_filename="Powell_singular_gradient_convergence.png")
    translate(input_filename="Powell_singular_gradient_descent_results.txt",
              output_filename="Powell_singular_residual_descent_results.txt",
              x_star=[0.0,0.0,0.0,0.0])
    draw_residual_convergence(filename="Powell_singular_residual_descent_results.txt",save_plot=True,
                              plot_filename="Powell_singular_residual_convergence.png")
    print("="*50)
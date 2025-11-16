import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.Descent_Method import conjugate_gradient_method
import numpy as np
from src.draw import draw_residual_convergence

def Hilbert_matrix(n):
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i, j] = 1.0 / (i + j + 1)
    return H

def main():
    for n in [5, 8 , 12 ,20]:
        print(f"\n当希尔伯特矩阵维度为 {n} 时：")
        H = Hilbert_matrix(n)
        b = np.ones(n)
        x0 = np.zeros(n)
        x, num_iter = conjugate_gradient_method(H, b, x0, tol=1e-6, max_iter=100,save_full_data=False, 
                       filename=f"question1_results_{n}.txt")
        print(f"最终迭代次数为： {num_iter}")
        print(f"最终解为： {x}")
        draw_residual_convergence(filename=f"question1_results_{n}.txt", 
                             save_plot=True, plot_filename=f"question1_residual_convergence_{n}.png")
        print("="*50)
        
if __name__ == "__main__":
    main()

import numpy as np
from .Line_Search import wolfe_search
import os
from pathlib import Path

def steepest_descent(f, grad_f, x0, tol=1e-5, max_iter=1000, save_full_data=False, 
                    filename="gradient_descent_results.txt"):
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    filepath = data_dir / filename
    
    x = x0.copy()
    
    with open(filepath, 'w') as file:
        if save_full_data:
            file.write("iteration,gradient_norm,x\n")
        else:
            file.write("iteration,gradient_norm\n")

        for i in range(max_iter):
            grad = grad_f(x)
            norm_grad = np.linalg.norm(grad)

            if save_full_data:
                file.write(f"{i},{norm_grad},{x.tolist()}\n")
            else:
                file.write(f"{i},{norm_grad}\n")
            
            if norm_grad < tol:
                return x, i
            
            direction = -grad 
            
            def f_alpha(alpha):
                return f(x + alpha * direction)
            
            def f_deriv_alpha(alpha):
                return np.dot(grad_f(x + alpha * direction), direction)
            
            alpha_opt = wolfe_search(f_alpha, f_deriv_alpha)
            x = x + alpha_opt * direction
    
    print("最大迭代次数达到，未收敛。")
    return x, max_iter

def pure_newton_method(f, grad_f, hess_f, x0, tol=1e-5, max_iter=1000, save_full_data=False, 
                       filename="newton_method_results.txt"):
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    filepath = data_dir / filename
    
    x = x0.copy()
    
    with open(filepath, 'w') as file:
        if save_full_data:
            file.write("iteration,gradient_norm,x\n")
        else:
            file.write("iteration,gradient_norm\n")
        for i in range(max_iter):
            grad = grad_f(x)
            norm_grad = np.linalg.norm(grad)

            if save_full_data:
                file.write(f"{i},{norm_grad},{x.tolist()}\n")
            else:
                file.write(f"{i},{norm_grad}\n")
            
            if norm_grad < tol:
                return x, i
            hess = hess_f(x)
            use_newton = True
            
            try:
                direction = -np.linalg.solve(hess, grad)
                if np.dot(direction, grad) >= 0:
                    use_newton = False
                    print("Hessian is not positive definite. Using gradient descent direction.")
            except np.linalg.LinAlgError:
                use_newton = False
                print("Hessian is singular or ill-conditioned. Using gradient descent direction.")
            if not use_newton:
                direction = -grad

            if use_newton:
                x = x + direction 
            else:
                def f_alpha(alpha):
                    return f(x + alpha * direction)
                
                def f_deriv_alpha(alpha):
                    return np.dot(grad_f(x + alpha * direction), direction)
                
                alpha_opt = wolfe_search(f_alpha, f_deriv_alpha)
                x = x + alpha_opt * direction
    
    print("最大迭代次数达到，未收敛。")
    return x, max_iter

def newton_method_with_linesearch(f, grad_f, hess_f, x0, tol=1e-5, max_iter=1000, 
                                 save_full_data=False, filename="newton_linesearch_results.txt"):
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    filepath = data_dir / filename
    
    x = x0.copy()
    
    with open(filepath, 'w') as file:
        if save_full_data:
            file.write("iteration,gradient_norm,x\n")
        else:
            file.write("iteration,gradient_norm\n")
        
        for i in range(max_iter):
            grad = grad_f(x)
            hess = hess_f(x)
            norm_grad = np.linalg.norm(grad)

            if save_full_data:
                file.write(f"{i},{norm_grad},{x.tolist()}\n")
            else:
                file.write(f"{i},{norm_grad}\n")
            
            if norm_grad < tol:
                return x, i
            
            try:
                direction = np.linalg.solve(hess, -grad)  
            except np.linalg.LinAlgError:
                print("Hessian is singular. Using gradient descent direction.")
                direction = -grad 
            else:
                if np.dot(direction, grad) >= 0:
                    print("Hessian is not positive definite. Using gradient descent direction.")
                    direction = -grad
            
            def f_alpha(alpha):
                return f(x + alpha * direction)
            
            def f_deriv_alpha(alpha):
                return np.dot(grad_f(x + alpha * direction), direction)
            
            alpha_opt = wolfe_search(f_alpha, f_deriv_alpha)
            x = x + alpha_opt * direction
    
    print("最大迭代次数达到，未收敛。")
    return x, max_iter

def Barzilar_Borwein_method(f, grad_f, x0, tol=1e-5, max_iter=1000, 
                           save_full_data=False, filename="barzilar_borwein_results.txt"):
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    filepath = data_dir / filename
    
    x = x0.copy()
    grad_prev = grad_f(x)
    x_prev = x0.copy()
    
    with open(filepath, 'w') as file:
    
        if save_full_data:
            file.write("iteration,gradient_norm,x\n")
        else:
            file.write("iteration,gradient_norm\n")
        
        for k in range(max_iter):
            grad = grad_f(x)
            norm_grad = np.linalg.norm(grad)
            
        
            if save_full_data:
                file.write(f"{k},{norm_grad},{x.tolist()}\n")
            else:
                file.write(f"{k},{norm_grad}\n")
            
            if norm_grad < tol:
                return x, k
                
            if k == 0:
                direction = -grad
                alpha = wolfe_search(lambda a: f(x + a * direction), 
                                   lambda a: np.dot(grad_f(x + a * direction), direction))
            else:
                s = x - x_prev
                y = grad - grad_prev
                if k % 2 == 0:
                    alpha = np.dot(s, s) / np.dot(s, y) 
                else:
                    alpha = np.dot(s, y) / np.dot(y, y)  

                if alpha <= 1e-10 or alpha > 1e10:
                    alpha = 1.0
                    
                direction = -grad  
                
            x_prev = x.copy()
            grad_prev = grad.copy()
            x = x + alpha * direction

    print("最大迭代次数达到，未收敛。")
    return x, max_iter

def conjugate_gradient_method(A, b, x0=None, tol=1e-5, max_iter=None,save_full_data=False, 
                       filename="conjugate_gradient_results.txt"):
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    filepath = data_dir / filename
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)
    else:
        x = x0.copy()
    r = A @ x - b
    p = -r.copy()
    rsold = r @ r
    if max_iter is None:
        max_iter = n+1
    with open(filepath, 'w') as file:
        if save_full_data:
            file.write("iteration,residual_norm,x\n")
        else:
            file.write("iteration,residual_norm\n")
        for i in range(max_iter):
            Ap = A @ p          
            alpha = rsold / (p @ Ap)
            x += alpha * p
            r += alpha * Ap
            rsnew = r @ r
            if save_full_data:
                file.write(f"{i},{np.sqrt(rsnew)},{x.tolist()}\n")
            else:
                file.write(f"{i},{np.sqrt(rsnew)}\n")
            if np.sqrt(rsnew) < tol:  
                return x, i+1
            beta = rsnew / rsold      
            p = -r + beta * p         
            rsold = rsnew
    return x, max_iter

def BFGS_method(f, grad_f, x0, tol=1e-6, max_iter=1000, save_full_data=False, 
         filename="BFGS_results.txt"):
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    filepath = data_dir / filename
    
    n = len(x0)
    x = x0.copy()
    H = np.eye(n)  
    
    with open(filepath, 'w') as file:
        if save_full_data:
            file.write("iteration,gradient_norm,x\n")
        else:
            file.write("iteration,gradient_norm\n")
        
        for i in range(max_iter):
            grad = grad_f(x)
            norm_grad = np.linalg.norm(grad)

            if save_full_data:
                file.write(f"{i},{norm_grad},{x.tolist()}\n")
            else:
                file.write(f"{i},{norm_grad}\n")
            
            if norm_grad < tol:
                return x, i
            
            direction = -H @ grad  
            
            def f_alpha(alpha):
                return f(x + alpha * direction)
            
            def f_deriv_alpha(alpha):
                return np.dot(grad_f(x + alpha * direction), direction)
            
            alpha_opt = wolfe_search(f_alpha, f_deriv_alpha)
            s = alpha_opt * direction  
            x_new = x + s
            y = grad_f(x_new) - grad  
            
            rho = 1.0 / (y @ s)
            I = np.eye(n)
            H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
            
            x = x_new
    
    print("最大迭代次数达到，未收敛。")
    return x, max_iter
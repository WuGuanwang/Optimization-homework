import numpy as np

def search_interval(f, alpha0=1, h=1e-5, t=2, max_iter=100):
    alpha = alpha0
    f_alpha = f(alpha)
    f_alpha_plus_h = f(alpha + h)
    
    if f_alpha_plus_h > f_alpha:
        # Search in the negative direction
        h = -h
    
    for _ in range(max_iter):
        f_alpha_plus_h = f(alpha + h)
        if f_alpha_plus_h >= f_alpha:
            if h > 0:
                return (alpha - h / t, alpha + h)
            else:
                return (alpha + h, alpha - h / t)
        
        alpha += h
        f_alpha = f_alpha_plus_h
        h *= t
    
    raise ValueError("Maximum iterations reached without finding an interval.")

def golden_section_search(f, tol=1e-5, max_iter=100):
    a,b=search_interval(f)
    phi = (np.sqrt(5)-1) / 2
    lamda = a + (1 - phi) * (b - a)
    mu = a + phi * (b - a)
    f_lamda = f(lamda)
    f_mu = f(mu)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            return (a + b) / 2
        if f_lamda < f_mu:
            b = mu
            mu = lamda
            f_mu = f_lamda
            lamda = a + (1 - phi) * (b - a)
            f_lamda = f(lamda)
        else:
            a = lamda
            lamda = mu
            f_lamda = f_mu
            mu = a + phi * (b - a)
            f_mu = f(mu)
    raise ValueError("Maximum iterations reached without finding a minimum.")

def goldstein_search(f,f_deriv_0,alpha=1,rho=0.1,t=2,max_iter=100):
    f_0 = f(0)
    alpha1=0
    alpha2=float('inf')
    for _ in range(max_iter):
        f_alpha = f(alpha)
        if f_alpha <= f_0 + rho * alpha * f_deriv_0 :
            if f_alpha >= f_0 + (1 - rho) * alpha * f_deriv_0:
                return alpha
            else:
                alpha1 = alpha
                if alpha2 == float('inf'):
                    alpha *= t
                else:
                    alpha = (alpha1 + alpha2) / 2
        else:
            alpha2 = alpha
            alpha = (alpha1 + alpha2) / 2
    raise ValueError("Maximum iterations reached without satisfying Goldstein conditions.")

def wolfe_search(f,f_deriv,alpha=2,rho=0.1,sigma=0.4,max_iter=100):
    f_0 = f(0)
    f_deriv_0 = f_deriv(0)
    alpha1=0
    alpha2=float('inf')
    for _ in range(max_iter):
        f_alpha = f(alpha)
        f_deriv_alpha = f_deriv(alpha)
        if f_alpha <= f_0 + rho * alpha * f_deriv_0 :
            if f_deriv_alpha >= sigma * f_deriv_0:
                return alpha
            else:
                alpha1 = alpha
                if alpha2 == float('inf'):
                    alpha *= 2
                else:
                    alpha = (alpha1 + alpha2) / 2
        else:
            alpha2 = alpha
            alpha = (alpha1 + alpha2) / 2
    raise ValueError("Maximum iterations reached without satisfying Wolfe conditions.")

def strong_wolfe_search(f,f_deriv,alpha=1,rho=0.1,sigma=0.4,max_iter=100):
    f_0 = f(0)
    f_deriv_0 = f_deriv(0)
    alpha1=0
    alpha2=float('inf')
    for _ in range(max_iter):
        f_alpha = f(alpha)
        f_deriv_alpha = f_deriv(alpha)
        if f_alpha <= f_0 + rho * alpha * f_deriv_0 :
            if abs(f_deriv_alpha) <= -sigma * f_deriv_0:
                return alpha
            else:
                alpha1 = alpha
                if alpha2 == float('inf'):
                    alpha *= 2
                else:
                    alpha = (alpha1 + alpha2) / 2
        else:
            alpha2 = alpha
            alpha = (alpha1 + alpha2) / 2
    raise ValueError("Maximum iterations reached without satisfying Strong Wolfe conditions.")

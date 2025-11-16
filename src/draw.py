import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def draw_gradient_convergence(filename="gradient_descent_results.txt", 
                             save_plot=True, plot_filename="gradient_convergence.png",
                             figsize=(10, 6), dpi=300):
    picture_dir = Path("picture")
    picture_dir.mkdir(exist_ok=True)
    plot_filepath = picture_dir / plot_filename

    data_dir = Path("data")
    filepath = data_dir / filename
    
    try:
        iterations = []
        gradient_norms = []
        
        with open(filepath, 'r') as file:
            lines = file.readlines()
            
            header = lines[0].strip().split(',')
            has_x_data = 'x' in header
           
            for line in lines[1:]:
                data = line.strip().split(',')
                iterations.append(int(data[0]))
                gradient_norms.append(float(data[1]))
       
        plt.figure(figsize=figsize)
        plt.semilogy(iterations, gradient_norms, 'b-', linewidth=2, label='norm of gradient')
        plt.xlabel('iterations', fontsize=12)
        plt.ylabel('norm of gradient (log scale)', fontsize=12)
        plt.title('Gradient Descent Convergence', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        if save_plot:
            plt.savefig(plot_filepath, dpi=dpi, bbox_inches='tight')
            print(f"图像已保存为 {plot_filepath}")
        
        plt.show()
       
    except FileNotFoundError:
        print(f"错误: 找不到文件 {filepath}")
    except Exception as e:
        print(f"读取文件时出错: {e}")

def draw_residual_convergence(filename="residual_descent_results.txt", 
                             save_plot=True, plot_filename="residual_convergence.png",
                             figsize=(10, 6), dpi=300):
    
    picture_dir = Path("picture")
    picture_dir.mkdir(exist_ok=True)
    plot_filepath = picture_dir / plot_filename

    data_dir = Path("data")
    filepath = data_dir / filename
    
    try:
        iterations = []
        residual_norms = []
        
        with open(filepath, 'r') as file:
            lines = file.readlines()
            
            header = lines[0].strip().split(',')
            has_x_data = 'x' in header
           
            for line in lines[1:]:
                data = line.strip().split(',')
                iterations.append(int(data[0]))
                residual_norms.append(float(data[1]))
       
        plt.figure(figsize=figsize)
        plt.semilogy(iterations, residual_norms, 'b-', linewidth=2, label='norm of residual')
        plt.xlabel('iterations', fontsize=12)
        plt.ylabel('norm of residual (log scale)', fontsize=12)
        plt.title('Residual Descent Convergence', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        if save_plot:
            plt.savefig(plot_filepath, dpi=dpi, bbox_inches='tight')
            print(f"图像已保存为 {plot_filepath}")
        
        plt.show()
       
    except FileNotFoundError:
        print(f"错误: 找不到文件 {filepath}")
    except Exception as e:
        print(f"读取文件时出错: {e}")

def translate(x_star,input_filename, output_filename):
    """
    读取梯度下降输出文件，计算每个迭代点与最优解的残差范数
    
    参数:
    input_filename: 输入文件名（在data目录下）
    x_star: 最优解，numpy数组
    output_filename: 输出文件名（将保存在data目录下）
    """
    x_star = np.array(x_star)
    input_filepath = Path("data") / input_filename
    output_filepath = Path("data") / output_filename

    try:
        with open(input_filepath, 'r') as infile:
            lines = infile.readlines()
       
        if len(lines) < 2:
            print(f"错误: 文件 {input_filepath} 内容不足")
            return False
            
        header = lines[0].strip().split(',')
        
        if 'x' not in header:
            print(f"错误: 文件 {input_filepath} 不包含x数据，无法计算残差")
            return False
        
        x_index = header.index('x')
        
        with open(output_filepath, 'w') as outfile:
            outfile.write("iteration,residual_norm\n")
            for line in lines[1:]:
                line = line.strip()
                start = line.find('[', line.find(',') * x_index)  
                end = line.find(']', start) + 1  
                
                iteration = int(line[:start-1].split(',')[0])
                x_str = line[start:end]
                
                x_current = np.array([float(val.strip()) for val in x_str.strip('[]').split(',')])
                residual_norm = np.linalg.norm(x_current - x_star)
                outfile.write(f"{iteration},{residual_norm}\n")
        return True
        
    except FileNotFoundError:
        print(f"错误: 找不到输入文件 {input_filepath}")
        return False
    except Exception as e:
        print(f"处理文件时出错: {e}")
        return False
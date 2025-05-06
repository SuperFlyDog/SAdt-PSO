import os
import random
import time
from statistics import mean, stdev
import csv

import numpy as np


def griewank_function(x):
    d = len(x)
    return 1 + np.sum(x ** 2 / 4000) - np.prod(np.cos(x / np.sqrt(np.arange(1, d + 1))))

def rastrigin_function(x):
    A = 10
    return A * len(x) + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))

class BBExpPSO:
    def __init__(self, N, D, M, lb, ub):
        """初始化BBExpPSO算法
        Args:
            N (int): 粒子数量
            D (int): 搜索空间维度
            M (int): 最大迭代次数
            lb (float): 搜索空间下界
            ub (float): 搜索空间上界
        """
        self.N = N  # 粒子数量
        self.D = D  # 搜索空间维度
        self.M = M  # 最大迭代次数
        self.lb = lb  # 搜索空间下界
        self.ub = ub  # 搜索空间上界

        # 初始化粒子位置和速度
        self.x = np.random.uniform(self.lb, self.ub, (self.N, self.D))
        self.v = np.zeros((self.N, self.D))

        # 初始化个体最优位置和全局最优位置
        self.pbest = self.x.copy()
        self.pbest_value = np.full(self.N, np.inf)
        self.gbest = np.zeros(self.D)
        self.gbest_value = np.inf

    def objective_function(self, x):
        return np.sum(x ** 2)

    def update_particles(self):
        """更新粒子的位置和速度"""
        for i in range(self.N):
            # 更新个体最优
            fitness = self.objective_function(self.x[i])
            if fitness < self.pbest_value[i]:
                self.pbest[i] = self.x[i]
                self.pbest_value[i] = fitness

            # 更新全局最优
            if fitness < self.gbest_value:
                self.gbest = self.x[i]
                self.gbest_value = fitness

        for i in range(self.N):
            for d in range(self.D):
                if random.random() < 0.5:
                    self.x[i][d] = self.pbest[i][d]
                else:
                    mean = (self.pbest[i][d] + self.gbest[d]) / 2
                    std = abs(self.pbest[i][d] - self.gbest[d])
                    self.x[i][d] = np.random.normal(mean, std)

            # 边界处理
            self.x[i] = np.clip(self.x[i], self.lb, self.ub)

    def optimize(self):
        """执行优化算法
        Returns:
            tuple: (全局最优值, 全局最优位置)
        """
        for _ in range(self.M):
            self.update_particles()
        return self.gbest_value, self.gbest


def calculate_reliability(std_time, std_error, w1=0.5, w2=0.5):
    """
    计算算法的稳定性指标 Reliability。

    Args:
        std_time (float): 时间的标准差
        std_error (float): 误差的标准差
        w1 (float): 时间权重（默认为0.5）
        w2 (float): 误差权重（默认为0.5）

    Returns:
        float: 算法稳定性指标 Reliability
    """
    reliability_time = 1 / std_time if std_time > 0 else float('inf')
    reliability_error = 1 / std_error if std_error > 0 else float('inf')
    return w1 * reliability_time + w2 * reliability_error


def log_results_to_csv(filename, algorithm_name, test_name, test_expression, parameters, mean_error, std_error,
                       mean_time):
    """
    将测试结果记录到 CSV 文件中。

    Args:
        filename (str): CSV 文件名
        algorithm_name (str): 算法名称
        test_name (str): 测试函数名
        test_expression (str): 测试函数表达式
        parameters (dict): 测试中使用的参数
        mean_error (float): 平均误差
        mean_time (float): 平均执行时间
        reliability (float): 稳定性指标
    """
    write_header = not os.path.exists(filename)

    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file,
                                fieldnames=["Algorithm", "Test Name", "Test Expression", "Parameters", "Mean Error",
                                            "Std Error", "Mean Time"])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "Algorithm": algorithm_name,
            "Test Name": test_name,
            "Test Expression": test_expression,
            "Parameters": str(parameters),
            "Mean Error": mean_error,
            "Std Error": std_error,
            "Mean Time": mean_time
        })


def perform_analysis_bbexp(n_runs, N, D, M, lb, ub):
    """
    执行多次BBExpPSO运行并分析其性能

    Args:
        n_runs (int): 运行次数
        N (int): 粒子数量
        D (int): 维度
        M (int): 最大迭代次数
        lb (float): 搜索空间下界
        ub (float): 搜索空间上界
    """
    empirical_errors = []
    execution_times = []

    for i in range(n_runs):
        start_time = time.time()

        # Run optimization
        bbexp_pso = BBExpPSO(N, D, M, lb, ub)
        best_value, _ = bbexp_pso.optimize()

        # Calculate empirical error
        empirical_error = abs(best_value - 0)
        empirical_errors.append(empirical_error)

        # Record execution time
        execution_time = time.time() - start_time
        execution_times.append(execution_time)

    # Calculate statistics
    mean_error = mean(empirical_errors)
    mean_time = mean(execution_times)

    if len(empirical_errors) > 1:
        std_error = stdev(empirical_errors)
        std_time = stdev(execution_times)
    else:
        std_error = 0.0
        std_time = 0.0

    # Calculate Reliability
    reliability = calculate_reliability(std_time, std_error, w1=0.5, w2=0.5)

    # Print analysis results
    print("\nPerformance Analysis Results:")
    print("-----------------------------")
    print(f"Number of Runs: {n_runs}")
    print(f"Mean Empirical Error: {mean_error:.6f}")
    print(f"Standard Deviation of Error: {std_error:.6f}")
    print(f"Mean Execution Time: {mean_time:.3f} seconds")
    print(f"Standard Deviation of Time: {std_time:.3f} seconds")

    # Log results to CSV
    algorithm_name = "BBExpPSO"
    test_name = "sphere"
    test_expression = "f(x) = 1 + (1 / 4000) * sum((R * (x - O))_i^2) - prod(cos((R * (x - O))_i / sqrt(i)))"

    parameters = {
        "N": N,
        "D": D,
        "M": M,
        "lb": lb,
        "ub": ub
    }
    log_results_to_csv("test_results_a.csv", algorithm_name, test_name, test_expression, parameters, mean_error, std_error,
                       mean_time)


if __name__ == '__main__':
    N = 30  # 粒子数量
    D = 30  # 维度
    M = 1000  # 最大迭代次数
    lb = -100 # 搜索空间下界
    ub = 100  # 搜索空间上界
    n_runs = 10  # 运行次数

    perform_analysis_bbexp(n_runs, N, D, M, lb, ub)

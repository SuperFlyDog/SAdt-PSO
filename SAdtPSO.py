import os

import numpy as np
import random
import matplotlib
import time
import csv

from statistics import mean, stdev

matplotlib.use('TkAgg')


# import matplotlib.pyplot as plt
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
    # 避免标准差为 0 导致除以 0 的情况
    reliability_time = 1 / std_time if std_time > 0 else float('inf')
    reliability_error = 1 / std_error if std_error > 0 else float('inf')

    # 计算加权指标
    reliability = w1 * reliability_time + w2 * reliability_error
    return reliability


def rastrigin_function(x):
    A = 10
    return A * len(x) + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))


def griewank_function(x):
    d = len(x)
    return 1 + np.sum(x ** 2 / 4000) - np.prod(np.cos(x / np.sqrt(np.arange(1, d + 1))))


def objective_function(x):
    shift = 600
    x = x - shift
    return griewank_function(x)

def metropolis_acceptance(delta_f, temperature):
    # print(temperature)
    """模拟退火的 Metropolis 准则"""
    if delta_f > 0:  # 如果适应值变好，直接接受
        return True
    else:  # 否则根据概率决定是否接受
        probability = np.exp(delta_f / temperature)
        return random.random() < probability


class BBExpPSO_model:
    def __init__(self, N, D, M, lb, ub, initial_temperature, eta, duration):
        # 初始化PSO模型

        self.N = N  # 粒子数量
        self.D = D  # 每个粒子的维度
        self.M = M  # 最大迭代次数
        self.lb = lb  # 搜索空间的下界
        self.ub = ub  # 搜索空间的上界
        self.initial_temperature = initial_temperature  # 模拟退火的初始温度
        self.eta = eta  # 全局适应值变化率的阈值，用于判断是否陷入“僵局”
        self.duration = duration  # 监测适应值变化的时长（滑动窗口大小）

        self.x = np.zeros((self.N, self.D))  # 粒子的位置矩阵
        self.pbest = np.zeros((self.N, self.D))  # 粒子的历史最优位置矩阵
        self.gbest = np.zeros((1, self.D))  # 全局最优位置
        self.p_fit = np.zeros(self.N)  # 每个粒子的适应值
        self.fit = 1e8  # 当前全局最优适应值
        self.theoretical_minimum = 0  # 理论最优解

        # 创建X-Von Neumann拓扑结构
        self.adjacency_matrix = self.x_von_neumann_topology()

    def init_pop(self):
        """初始化粒子群"""
        for i in range(self.N):
            self.x[i] = np.random.uniform(self.lb, self.ub, self.D)  # 随机生成粒子的位置
            self.pbest[i] = self.x[i]  # 初始化每个粒子的个体最优位置为其初始位置
            aim = objective_function(self.x[i])  # 计算粒子的适应值
            self.p_fit[i] = aim
            if aim < self.fit:  # 如果当前粒子优于全局最优
                self.fit = aim
                self.gbest = self.x[i]  # 更新全局最优位置

    def x_von_neumann_topology(self):
        """
        构建X-Von Neumann拓扑连接矩阵，支持非完全平方数粒子数
        :return: X-Von Neumann拓扑连接矩阵
        """
        side_length = int(np.ceil(np.sqrt(self.N)))  # 近似网格边长
        adjacency_matrix = np.zeros((self.N, self.N))

        for i in range(self.N):
            row, col = divmod(i, side_length)  # 当前粒子的网格坐标
            neighbors = []

            # 上下左右邻居
            if row > 0:  # 上
                neighbors.append((row - 1) * side_length + col)
            if row < side_length - 1:  # 下
                down = (row + 1) * side_length + col
                if down < self.N:  # 确保不越界
                    neighbors.append(down)
            if col > 0:  # 左
                neighbors.append(row * side_length + (col - 1))
            if col < side_length - 1:  # 右
                right = row * side_length + (col + 1)
                if right < self.N:  # 确保不越界
                    neighbors.append(right)

            # 对角线邻居
            if row > 0 and col > 0:  # 左上
                neighbors.append((row - 1) * side_length + (col - 1))
            if row > 0 and col < side_length - 1:  # 右上
                right_up = (row - 1) * side_length + (col + 1)
                if right_up < self.N:  # 确保不越界
                    neighbors.append(right_up)
            if row < side_length - 1 and col > 0:  # 左下
                left_down = (row + 1) * side_length + (col - 1)
                if left_down < self.N:  # 确保不越界
                    neighbors.append(left_down)
            if row < side_length - 1 and col < side_length - 1:  # 右下
                right_down = (row + 1) * side_length + (col + 1)
                if right_down < self.N:  # 确保不越界
                    neighbors.append(right_down)

            # 更新邻接矩阵
            for neighbor in neighbors:
                adjacency_matrix[i][neighbor] = 1
                adjacency_matrix[neighbor][i] = 1  # 保持对称性

        return adjacency_matrix

    def update_topology(self):
        """根据逻辑重连拓扑"""
        new_adjacency_matrix = self.adjacency_matrix.copy()

        # 计算粒子间距离矩阵
        distances = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    distances[i][j] = np.linalg.norm(self.x[i] - self.x[j])

        # 找到彼此距离最近的两个粒子
        min_distance = np.inf
        min_pair = (0, 0)
        for i in range(self.N):
            for j in range(self.N):
                if distances[i][j] < min_distance and new_adjacency_matrix[i][j] == 1:
                    min_distance = distances[i][j]
                    min_pair = (i, j)

        # 切断它们之间的连接
        i, j = min_pair
        new_adjacency_matrix[i][j] = 0
        new_adjacency_matrix[j][i] = 0

        # 找到距离最远的另一个粒子
        max_distance = -np.inf
        max_particle = -1
        for k in range(self.N):
            if k != i and k != j:
                dist_to_k = max(distances[i][k], distances[j][k])
                if dist_to_k > max_distance:
                    max_distance = dist_to_k
                    max_particle = k

        # 将其中一个粒子单向连接到最远的粒子
        chosen_particle = random.choice([i, j])
        new_adjacency_matrix[chosen_particle][max_particle] = 1  # 单向连接

        return new_adjacency_matrix

    def update(self):

        """粒子更新"""
        delta_f_best_window = []  # 窗口存储Δf_best的值

        for t in range(self.M):
            previous_best_fit = self.fit  # 保存当前全局最优值
            for i in range(self.N):
                for d in range(self.D):
                    if random.random() < 0.5:  # 按概率选择个体经验
                        self.x[i][d] = self.pbest[i][d]
                    else:  # 使用局部邻居的经验更新
                        neighbors = np.where(self.adjacency_matrix[i] == 1)[0]  # 获取邻居索引
                        if len(neighbors) > 0:
                            local_best_idx = neighbors[np.argmin(self.p_fit[neighbors])]  # 找到局部最优粒子索引
                            local_best = self.pbest[local_best_idx]  # 获取局部最优位置
                            mean = (local_best[d] + self.pbest[i][d]) / 2
                            std = abs(local_best[d] - self.pbest[i][d])
                        else:  # 如果无邻居，使用全局经验
                            mean = (self.pbest[i][d] + self.gbest[d]) / 2
                            std = abs(self.pbest[i][d] - self.gbest[d])
                        self.x[i][d] = np.random.normal(mean, std)

                # 边界处理
                self.x[i] = np.clip(self.x[i], self.lb, self.ub)
                aim = objective_function(self.x[i])
                if aim < self.p_fit[i]:
                    self.p_fit[i] = aim
                    self.pbest[i] = self.x[i]
                    if self.p_fit[i] < self.fit:
                        self.gbest = self.x[i]
                        self.fit = self.p_fit[i]

            # 更新适应值变化窗口
            delta_f = abs(self.fit - previous_best_fit)
            delta_f_best_window.append(delta_f)
            if len(delta_f_best_window) > self.duration:
                delta_f_best_window.pop(0)

            mean_delta_f_best = np.mean(delta_f_best_window)  # 计算窗口内的平均变化
            temperature = self.initial_temperature * (1 - t / self.M)  # 更新温度

            # 判断是否需要拓扑重连
            if mean_delta_f_best < self.eta:
                # 保存当前拓扑
                old_adjacency_matrix = self.adjacency_matrix.copy()

                # 进行拓扑重连
                self.adjacency_matrix = self.update_topology()

                # 测试新拓扑的效果
                temp_delta_f = []
                for _ in range(10):
                    for i in range(self.N):
                        for d in range(self.D):
                            if random.random() < 0.5:
                                self.x[i][d] = self.pbest[i][d]
                            else:
                                neighbors = np.where(self.adjacency_matrix[i] == 1)[0]
                                if len(neighbors) > 0:
                                    local_best_idx = neighbors[np.argmin(self.p_fit[neighbors])]
                                    local_best = self.pbest[local_best_idx]
                                    mean = (local_best[d] + self.pbest[i][d]) / 2
                                    std = abs(local_best[d] - self.pbest[i][d])
                                else:
                                    mean = (self.pbest[i][d] + self.gbest[d]) / 2
                                    std = abs(self.pbest[i][d] - self.gbest[d])
                                self.x[i][d] = np.random.normal(mean, std)

                            self.x[i] = np.clip(self.x[i], self.lb, self.ub)
                            aim = objective_function(self.x[i])
                            if aim < self.p_fit[i]:
                                self.p_fit[i] = aim
                                self.pbest[i] = self.x[i]
                                if self.p_fit[i] < self.fit:
                                    self.gbest = self.x[i]
                                    self.fit = self.p_fit[i]

                    temp_delta_f.append(abs(self.fit - previous_best_fit))
                # Todo 为什么每次都会接受新拓扑？ 可能原因1：参数设置不当。 可能原因2：代码有问题？
                # Todo 原因：metropolis_acceptance->mean_temp_delta_f - mean_delta_f_best过于小
                mean_temp_delta_f = np.mean(temp_delta_f)  # 拓扑重连后进行测试迭代时，适应值变化率的平均值。
                if metropolis_acceptance(mean_temp_delta_f - mean_delta_f_best, temperature):
                    # print(f"Iteration {t}: New topology accepted.")
                    nothing = 1
                else:
                    print(f"Iteration {t}: Reverting to old topology.")
                    self.adjacency_matrix = old_adjacency_matrix

        return self.fit


def perform_analysis(n_runs, N, D, M, lb, ub, initial_temperature, eta, duration):
    """
    执行多次PSO运行并分析其性能

    Args:
        n_runs (int): 运行次数
        N (int): 粒子数量
        D (int): 维度
        M (int): 最大迭代次数
        lb (float): 搜索空间下界
        ub (float): 搜索空间上界
        initial_temperature (float): 模拟退火初始温度
        eta (float): 适应值变化率阈值
        duration (int): 监测适应值变化的时长
    """
    empirical_errors = []
    execution_times = []

    for i in range(n_runs):
        start_time = time.time()

        # Run optimization
        bbexp_pso = BBExpPSO_model(N, D, M, lb, ub, initial_temperature, eta, duration)
        bbexp_pso.init_pop()
        gbest_value = bbexp_pso.update()

        # Calculate empirical error
        empirical_error = abs(gbest_value - bbexp_pso.theoretical_minimum)
        empirical_errors.append(empirical_error)

        # Record execution time
        execution_time = time.time() - start_time
        execution_times.append(execution_time)

    # Calculate statistics
    mean_error = mean(empirical_errors)
    mean_time = mean(execution_times)

    # Check if we have enough data points for standard deviation
    if len(empirical_errors) > 1:
        std_error = stdev(empirical_errors)
        std_time = stdev(execution_times)
    else:
        std_error = 0.0
        std_time = 0.0
    # 计算 Reliability
    reliability = calculate_reliability(std_time, std_error, w1=0.5, w2=0.5)  # 权重可调整
    # print(f"Reliability: {reliability:.4f}")

    # Print analysis results
    print("\nPerformance Analysis Results:")
    print("-----------------------------")
    print(f"Number of Runs: {n_runs}")
    print(f"Mean Empirical Error: {mean_error:.6f}")
    print(f"Standard Deviation of Error: {std_error:.6f}")
    print(f"Mean Execution Time: {mean_time:.3f} seconds")
    print(f"Standard Deviation of Time: {std_time:.3f} seconds")
    # 测试结果
    algorithm_name = "30-SAdt-PSO"
    test_name = "shifted_rotated_griewank_function"
    test_expression = ""

    parameters = {
        "N": N,
        "D": D,
        "M": M,
        "lb": lb,
        "ub": ub,
        "initial_temperature": initial_temperature,
        "eta": eta,
        "duration": duration
    }
    # 记录到 CSV 文件
    log_results_to_csv("test_results_a.csv", algorithm_name, test_name, test_expression, parameters, mean_error,
                       std_error,
                       mean_time)


if __name__ == '__main__':
    N = 30  # 粒子数量
    D = 30  # 维度
    M = 1000  # 最大迭代次数
    lb = -600  # 搜索空间下界
    ub = 600  # 搜索空间上界
    initial_temperature = 0.00001  # 初始温度
    eta = 0.0001  # 全局最优解的适应值变化率（Δfbest）阈值
    n_runs = 10  # 运行次数
    duration = 10  # 监测适应值变化的时长

    perform_analysis(n_runs, N, D, M, lb, ub, initial_temperature, eta, duration)

import numpy as np
import random
import matplotlib
import time
from statistics import mean, stdev

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class BBExpPSO_model:
    def __init__(self, N, D, M, lb, ub, initial_temperature, eta):

        self.N = N  # 粒子数量
        self.D = D  # 每个粒子的维度
        self.M = M  # 最大迭代次数
        self.lb = lb  # 下界
        self.ub = ub  # 上界
        self.initial_temperature = initial_temperature  # 初始温度
        self.eta = eta  # 阈值

        self.x = np.zeros((self.N, self.D))  # 粒子位置
        self.pbest = np.zeros((self.N, self.D))  # 粒子的历史最优位置
        self.gbest = np.zeros((1, self.D))  # 全局最优位置
        self.p_fit = np.zeros(self.N)  # 粒子的适应度
        self.fit = 1e8  # 最优适应度
        self.theoretical_minimum = 0  # 目标函数的理论最小值

        # 创建X-Von Neumann拓扑
        self.adjacency_matrix = self.x_von_neumann_topology()

    def objective_function(self, x):
        """目标函数: Rastrigin函数"""
        A = 10
        x1 = x[0]
        x2 = x[1]
        Z = 2 * A + x1 ** 2 - A * np.cos(2 * np.pi * x1) + x2 ** 2 - A * np.cos(2 * np.pi * x2)
        return Z

    def init_pop(self):
        """初始化粒子群"""
        for i in range(self.N):
            self.x[i] = np.random.uniform(self.lb, self.ub, self.D)  # 初始化粒子位置
            self.pbest[i] = self.x[i]  # 初始化个体最优位置
            aim = self.objective_function(self.x[i])
            self.p_fit[i] = aim
            if aim < self.fit:
                self.fit = aim
                self.gbest = self.x[i]

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

    def metropolis_acceptance(self, delta_f, temperature):
        """模拟退火的 Metropolis 准则"""
        if delta_f > 0:
            return True
        else:
            probability = np.exp(delta_f / temperature)
            return random.random() < probability

    def update_topology(self):
        """根据逻辑重连拓扑"""
        new_adjacency_matrix = self.adjacency_matrix.copy()

        # 计算粒子间距离矩阵
        distances = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    distances[i][j] = np.linalg.norm(self.x[i] - self.x[j])

        # 找到彼此最近的两个粒子
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

        # 将其中一个粒子连接到最远的粒子
        chosen_particle = random.choice([i, j])
        new_adjacency_matrix[chosen_particle][max_particle] = 1
        new_adjacency_matrix[max_particle][chosen_particle] = 1

        return new_adjacency_matrix

    def update(self):
        """粒子更新"""
        delta_f_best = []
        reconnected = False

        for t in range(self.M):
            previous_best_fit = self.fit

            for i in range(self.N):
                for d in range(self.D):
                    if random.random() < 0.5:
                        self.x[i][d] = self.pbest[i][d]
                    else:
                        neighbors = self.adjacency_matrix[i]  # 使用拓扑获取邻居
                        mean = (self.pbest[i][d] + self.gbest[d]) / 2
                        std = abs(self.pbest[i][d] - self.gbest[d])
                        self.x[i][d] = np.random.normal(mean, std)

                # 边界处理
                self.x[i] = np.clip(self.x[i], self.lb, self.ub)
                aim = self.objective_function(self.x[i])
                if aim < self.p_fit[i]:
                    self.p_fit[i] = aim
                    self.pbest[i] = self.x[i]
                    if self.p_fit[i] < self.fit:
                        self.gbest = self.x[i]
                        self.fit = self.p_fit[i]

            # 检查Δf_best是否小于阈值
            delta_f = abs(self.fit - previous_best_fit)
            delta_f_best.append(delta_f)

            # 计算当前温度
            temperature = self.initial_temperature * (1 - t / self.M)
            print('Temperature: ', temperature)
            if not reconnected and delta_f < self.eta:
                # 保存当前拓扑
                old_adjacency_matrix = self.adjacency_matrix.copy()

                # 进行拓扑重连
                self.adjacency_matrix = self.update_topology()
                reconnected = True

                # 测试重连后的Δf_best
                temp_delta_f = []
                for _ in range(10):  # 假设运行10次迭代
                    for i in range(self.N):
                        for d in range(self.D):
                            if random.random() < 0.5:
                                self.x[i][d] = self.pbest[i][d]
                            else:
                                neighbors = self.adjacency_matrix[i]
                                mean = (self.pbest[i][d] + self.gbest[d]) / 2
                                std = abs(self.pbest[i][d] - self.gbest[d])
                                self.x[i][d] = np.random.normal(mean, std)

                        # 边界处理
                        self.x[i] = np.clip(self.x[i], self.lb, self.ub)
                        aim = self.objective_function(self.x[i])
                        if aim < self.p_fit[i]:
                            self.p_fit[i] = aim
                            self.pbest[i] = self.x[i]
                            if self.p_fit[i] < self.fit:
                                self.gbest = self.x[i]
                                self.fit = self.p_fit[i]

                    temp_delta_f.append(abs(self.fit - previous_best_fit))

                # 判断是否接受新拓扑
                mean_delta_f_new = np.mean(temp_delta_f)
                if self.metropolis_acceptance(mean_delta_f_new - np.mean(delta_f_best), temperature):
                    print("接受新拓扑")
                else:
                    print("回滚到旧拓扑")
                    self.adjacency_matrix = old_adjacency_matrix

        return self.fit


def perform_analysis(n_runs, N, D, M, lb, ub, initial_temperature, eta):
    """
    Perform multiple runs of the BBExp PSO algorithm and analyze its performance.

    Args:
        n_runs (int): Number of runs to perform
        N (int): Population size
        D (int): Dimension
        M (int): Maximum iterations
        lb (float): Lower bound of search space
        ub (float): Upper bound of search space
        initial_temperature (float): Initial temperature for simulated annealing
        eta (float): Threshold for topology reconnection
    """
    empirical_errors = []
    execution_times = []

    for i in range(n_runs):
        start_time = time.time()

        # Run optimization
        bbexp_pso = BBExpPSO_model(N, D, M, lb, ub, initial_temperature, eta)
        bbexp_pso.init_pop()
        gbest_value = bbexp_pso.update()

        # Calculate empirical error
        empirical_error = abs(gbest_value - bbexp_pso.theoretical_minimum)
        empirical_errors.append(empirical_error)

        # Record execution time
        execution_time = time.time() - start_time
        execution_times.append(execution_time)

        print(f"Run {i + 1}/{n_runs}:")
        print(f"Empirical Error: {empirical_error:.6f}")
        print(f"Execution Time: {execution_time:.3f} seconds\n")

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

    # Print analysis results
    print("\nPerformance Analysis Results:")
    print("-----------------------------")
    print(f"Number of Runs: {n_runs}")
    print(f"Mean Empirical Error: {mean_error:.6f}")
    print(f"Standard Deviation of Error: {std_error:.6f}")
    print(f"Mean Execution Time: {mean_time:.3f} seconds")
    print(f"Standard Deviation of Time: {std_time:.3f} seconds")


if __name__ == '__main__':
    # Set parameters
    N = 30  # population size
    D = 2  # dimension
    M = 200  # maximum iterations
    lb = -5.0  # lower bound of search space
    ub = 5.0  # upper bound of search space
    initial_temperature = 3.0  # initial temperature for simulated annealing
    eta = 1e-5  # threshold for topology reconnection
    n_runs = 1  # number of runs for analysis

    perform_analysis(n_runs, N, D, M, lb, ub, initial_temperature, eta)

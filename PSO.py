import numpy as np
import random
import matplotlib
import time
from statistics import mean, stdev

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class BBExpPSO_model:
    def __init__(self, N, D, M):

        self.N = N  # 粒子数量
        self.D = D  # 每个粒子的维度
        self.M = M  # 最大迭代次数
        self.x = np.zeros((self.N, self.D))  # 粒子位置
        self.pbest = np.zeros((self.N, self.D))  # 粒子的历史最优位置
        self.gbest = np.zeros((1, self.D))  # 全局最优位置
        self.p_fit = np.zeros(self.N)  # 粒子的适应度
        self.fit = 1e8  # 最优适应度
        self.lb = -5.0  # 下界
        self.ub = 5.0  # 上界
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

    def update(self):
        """粒子更新"""
        for t in range(self.M):
            for i in range(self.N):
                for d in range(self.D):
                    if random.random() < 0.5:
                        self.x[i][d] = self.pbest[i][d]
                    else:
                        neighbors = self.adjacency_matrix[i]  # 使用X-Von Neumann拓扑获取邻居
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

        return self.fit

    def plot_particles(self):
        """绘制粒子分布图"""
        x = np.linspace(self.lb, self.ub, 100)
        y = np.linspace(self.lb, self.ub, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        for i in range(len(x)):
            for j in range(len(y)):
                Z[i, j] = self.objective_function([X[i, j], Y[i, j]])

        plt.figure(figsize=(10, 8))
        plt.contour(X, Y, Z, levels=20, cmap='viridis')
        plt.colorbar(label='Objective Function Value')
        plt.scatter(self.x[:, 0], self.x[:, 1], c='red', marker='o', label='Particles')
        plt.scatter(self.gbest[0], self.gbest[1], c='yellow', marker='*', s=200, label='Global Best')
        plt.title('BBExp PSO Particle Distribution (Final Iteration)')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.grid(True)
        plt.savefig('bbexp_pso_distribution.png')
        plt.show()


def perform_analysis(n_runs, N, D, M):
    """
    Perform multiple runs of the BBExp PSO algorithm and analyze its performance.

    Args:
        n_runs (int): Number of runs to perform
        N (int): Population size
        D (int): Dimension
        M (int): Maximum iterations
    """
    empirical_errors = []
    execution_times = []

    for i in range(n_runs):
        start_time = time.time()

        # Run optimization
        bbexp_pso = BBExpPSO_model(N, D, M)
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
    std_error = stdev(empirical_errors)
    mean_time = mean(execution_times)
    std_time = stdev(execution_times)

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
    N = 10  # population size
    D = 30  # dimension
    M = 200  # maximum iterations
    n_runs = 30  # number of runs for analysis

    perform_analysis(n_runs, N, D, M)
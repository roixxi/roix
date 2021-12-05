
import numpy as np

class Simplex(object):
    def __init__(self, obj, max_mode=False):
        self.max_mode = max_mode  # 默认是求min，如果是max目标函数要乘-1
        self.mat = np.array([[0] + obj]) * (-1 if max_mode else 1)      #矩阵先加入目标函数

    def add_constraint(self, a, b):
        self.mat = np.vstack([self.mat, [b] + a])      #矩阵加入约束

    def solve(self):
        m, n = self.mat.shape  # 矩阵里有1行目标函数，m - 1行约束，应加入m-1个松弛变量
        temp, B = np.vstack([np.zeros((1, m - 1)), np.eye(m - 1)]), list(range(n - 1, n + m - 1))  # temp是一个对角矩阵，B是个递增序列
        mat = self.mat = np.hstack([self.mat, temp])  # 横向拼接
        while mat[0, 1:].min() < 0:   #判断目标函数里是否还有负系数项
            col = np.where(mat[0, 1:] < 0)[0][0] + 1  # 在目标函数里找到第一个负系数的变量，找到替入变量
            row = np.array([mat[i][0] / mat[i][col] if mat[i][col] > 0 else 0x7fffffff for i in
                            range(1, mat.shape[0])]).argmin() + 1  # 找到最严格约束的行，也就找到替出变量
            if mat[row][col] <= 0: return None  # 若无替出变量，原问题无界
            mat[row] /= mat[row][col]    #替入变量和替出变量互换
            ids = np.arange(mat.shape[0]) != row
            mat[ids] -= mat[row] * mat[ids, col:col + 1]  # 对i!= row的每一行约束条件，做替入和替出变量的替换
            B[row] = col  #用B数组记录替入的替入变量
        return mat[0][0] * (1 if self.max_mode else -1), {B[i]: mat[i, 0] for i in range(1, m) if B[i] < n} #返回目标值，对应x的解就是基本变量为对应的bi，非基本变量为0


"""
       minimize -x1 - 14x2 - 6x3
       st
        x1 + x2 + x3 <=4
        x1 <= 2
        x3 <= 3
        3x2 + x3 <= 6
        x1 ,x2 ,x3 >= 0
       answer :-32
    """
t = Simplex([-1, -14, -6])
t.add_constraint([1, 1, 1], 4)
t.add_constraint([1, 0, 0], 2)
t.add_constraint([0, 0, 1], 3)
t.add_constraint([0, 3, 1], 6)
print(t.solve())
print(t.mat)
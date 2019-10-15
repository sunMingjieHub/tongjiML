import numpy as np


a = np.array([(1, 1, 2, 4, 2), (1, 3, 3, 4, 4)])
print("原矩阵：")
print(a)
t = np.mean(a, axis=1)
# m行
m = a.shape[0]
# n列
n = a.shape[1]

for i in range(m):
    for j in range(n):
        a[i][j] = a[i][j] - t[i]
print("进行平均值平滑处理后：")
print(a)

C = np.dot(a, a.T)
C = np.divide(C, n)
print("协方差矩阵: ")
print(C)
value, vector = np.linalg.eig(C)
print("特征值：")
print(value)
print("特征向量：")
print(vector)

P = vector.T
Y = np.dot(P[0], a)
print("最终降维后矩阵：")
print(Y)

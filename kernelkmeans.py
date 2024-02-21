# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import pairwise
import matplotlib.pyplot as plt

class KernelKMeans(object):

    def __init__(self,
                 n_clusters=8,
                 max_iter=300,
                 kernel=pairwise.linear_kernel):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.kernel = kernel

    def _initialize_cluster(self, X):
        self.N = np.shape(X)[0]
        self.y = np.random.randint(low=0, high=self.n_clusters, size=self.N)
        self.K = self.kernel(X)

    def fit_predict(self, X):
        self._initialize_cluster(X)
        for _ in range(self.max_iter):
            obj = np.tile(np.diag(self.K).reshape((-1, 1)), self.n_clusters)
            N_c = np.bincount(self.y)
            for c in range(self.n_clusters):
                obj[:, c] -= 2 * \
                    np.sum((self.K)[:, self.y == c], axis=1) / N_c[c]
                obj[:, c] += np.sum((self.K)[self.y == c][:, self.y == c]) / \
                    (N_c[c] ** 2)
            self.y = np.argmin(obj, axis=1)
        return self.y







def make_dataset(N):

    # data = np.random.rand(100, 2)
    X  = np.random.rand(N, 2) # 生成一个N行2列的二维数组，用于存放样本点
    # X[: N / 2, 0] = np.random.randn(N / 2)
    # X[N / 2:, 0] = np.random.randn(N / 2)
    # X[: N / 2, 1] = np.random.randn(N / 2)
    # X[N / 2:, 1] = np.random.randn(N / 2)
    return X


if __name__ == '__main__':

    X = make_dataset(500)

    # kernel k-means with linear kernel
    kkm_linear = KernelKMeans(
        n_clusters=8, max_iter=100, kernel=pairwise.linear_kernel)
    y_linear = kkm_linear.fit_predict(X)

    # kernel k-means with rbf kernel
    kkm_rbf = KernelKMeans(
        n_clusters=2, max_iter=100,
        kernel=lambda X: pairwise.rbf_kernel(X, gamma=0.1))
    y_rbf = kkm_rbf.fit_predict(X)

    plt.subplot(121)
    plt.scatter(X[y_linear == 0][:, 0], X[y_linear == 0][:, 1], c="blue")
    plt.scatter(X[y_linear == 1][:, 0], X[y_linear == 1][:, 1], c="red")
    plt.title("linear kernel")
    plt.axis("scaled")
    plt.subplot(122)
    plt.scatter(X[y_rbf == 0][:, 0], X[y_rbf == 0][:, 1], c="blue")
    plt.scatter(X[y_rbf == 1][:, 0], X[y_rbf == 1][:, 1], c="red")
    plt.title("rbf kernel")
    plt.axis("scaled")
    plt.savefig("/data/users/gaoli/exp_Robust/figures/11kernel_kmeans.png")
    plt.show()
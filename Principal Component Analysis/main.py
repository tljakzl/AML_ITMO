# coding=utf-8
import numpy as np
from sklearn.decomposition import PCA
from sklearn import datasets
import matplotlib
import matplotlib.pyplot as plt

# %matplotlib inline

X = datasets.load_digits().data  # Объекты
y = datasets.load_digits().target  # Отклики
print(X[0])
plt.imshow(X[0].reshape([8, 8]), cmap='Greys_r')
plt.show()

pca = PCA(n_components=2, svd_solver='full') #Создание объекта класса PCA. В качестве параметров выступает количество ГК и метод оптимизации
X_transformed = pca.fit(X).transform(X) #X_transformed -- ndarray объектов, где каждый объект описывается двумя ГК

plt.scatter(X_transformed[:101, 0], X_transformed[:101, 1], c=y[:101], edgecolor='none', s=40,cmap='winter')
plt.show()
plt.plot(X_transformed[:101, 0], X_transformed[:101, 1], 'o', markerfacecolor='red', markeredgecolor='k', markersize=8)
plt.show()

pca = PCA(n_components=64, svd_solver='full')
X_full = pca.fit(X).transform(X)
explained_variance = np.round(np.cumsum(pca.explained_variance_ratio_),3)
print(explained_variance)
plt.plot(np.arange(64), explained_variance, ls = '-')
plt.show()
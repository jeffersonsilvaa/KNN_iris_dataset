#teste com Iris dataset e kNN
#https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

dataset = np.loadtxt("iris.txt", delimiter=",")
print(dataset.shape)

X = dataset[:,0:3]
y = dataset[:,4]


knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=3, p=2,
           weights='uniform')

knn.fit(X, y)
knn.predict(X)
acuracia = knn.score(X, y)

print('-----------------------------------------------------------')
print('Acc = ', acuracia)
print('-----------------------------------------------------------')
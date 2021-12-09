# import các thư viện và data vào Python
import numpy as np
import matplotlib.pyplot as plt
import sys
print(sys.version)
from sklearn import neighbors, datasets

# Load dữ liệu và hiện thị vài dữ liệu mẫu. Các class được gán nhãn là 0, 1, và 2.
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
print ("Number of classes: %d" %len(np.unique(iris_y)))
print ('Number of data points: %d' %len(iris_y))

X0 = iris_X[iris_y == 0,:]
print ('\nSamples from class 0:\n', X0[:5,:])

X1 = iris_X[iris_y == 1,:]
print ('\nSamples from class 1:\n', X1[:5,:])

X2 = iris_X[iris_y == 2,:]
print ("\nSamples from class 2:\n", X2[:5,:])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
     iris_X, iris_y, test_size=50)

# Tách training và test sets
print ("Training size: %d" %len(y_train))
print ("Test size    : %d" %len(y_test))
clf = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# xét trường hợp đơn giản K = 1, tức là với mỗi điểm test data ta chỉ xét 1
# điểm training data gần nhất và lấy label của điểm đó để dự đoán cho điểm test này
print ("Print results for first 20 test data points:")
print ("Predicted labels: ", y_pred[20:40])
print ("Ground truth    : ", y_test[20:40])
clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Phương pháp đánh giá
from sklearn.metrics import accuracy_score
# Xét 1 điểm gần nhất
print ("Accuracy of 1NN: %.2f %%" %(100*accuracy_score(y_test, y_pred)))
# Xét 10 điểm gần nhất
clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print ("Accuracy of 10NN with major voting: %.2f %%" %(100*accuracy_score(y_test, y_pred)))
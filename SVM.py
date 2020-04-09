import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report as report
from sklearn.model_selection import GridSearchCV, train_test_split


#Import data from file
data, x, y = [], [], []
for line in open('file', 'r'):
    if '?' not in line:
        x.append(tuple(map(float, line.strip().split(',')[1:])))
        y.append(int(line.strip().split(',')[0]))
        data.append((*x,y))

#Class-based stuff
targets = tuple(set(y))
names = tuple(map(str, targets))
colors = ('r','g','b','c','m','y','k')

#Data-based stuff
x = StandardScaler().fit_transform(x)
pc = PCA(n_components=2).fit_transform(x)
x_train, x_test, y_train, y_test, px_train, px_test = train_test_split(x, y, pc)

#SVM parameters. Gamma search takes a lot of time
clf = GridSearchCV(SVC(), {'kernel':('linear', 'rbf', 'poly', 'sigmoid'),
                           'C':[10**_ for _ in range(-5,6)],
##                           'gamma':[10**_ for _ in range(6)],
                           'degree':[_ for _ in range(2,6)]
                           }, n_jobs=-2)

#SVM for original data
clf.fit(x_train, y_train)
y_train_pred = clf.predict(x_train)
y_test_pred = clf.predict(x_test)
y_pred = clf.predict(x)
print('SVM for original data:')
print('Chosen estimator: ', clf.best_params_)
print('Number of vectors for each class: ', clf.best_estimator_.n_support_)
print('Metrics for training sample:\n',
      report(y_train, y_train_pred,target_names=names, zero_division=1))
print('Metrics for test sample:\n',
      report(y_test, y_test_pred, target_names=names, zero_division=1))
print('Metrics for all data:\n',
      report(y, y_pred, target_names=names, zero_division=1), end='\n-----\n\n')

#SVM for 2 primary components (easy to visualize)
clf.fit(px_train, y_train)
y_train_pred = clf.predict(px_train)
y_test_pred = clf.predict(px_test)
y_pred = clf.predict(pc)
print('SVM for 2 primary components:')
print('Chosen estimator: ', clf.best_params_)
print('Number of vectors for each class: ', clf.best_estimator_.n_support_)
print('Metrics for training sample:\n',
      report(y_train, y_train_pred,target_names=names, zero_division=1))
print('Metrics for test sample:\n',
      report(y_test, y_test_pred, target_names=names, zero_division=1))
print('Metrics for all data:\n',
      report(y, y_pred, target_names=names, zero_division=1), end='\n-----\n\n')

#Plot predicted model
plt.subplot(1, 2, 1)
plt.title('Predicted for PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.axis('off')
for target, color in zip(targets,colors):
    plt.scatter([pc[row][0] for row in range(len(y))
                 if y_pred[row] == target],
                [pc[row][1] for row in range(len(y))
                 if y_pred[row] == target], c = color)
xlim = plt.xlim()
ylim = plt.ylim()
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 1000),
                     np.linspace(ylim[0], ylim[1], 1000))
xy = np.vstack([xx.ravel(), yy.ravel()]).T
Z = clf.best_estimator_.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.contour(xx, yy, Z, colors='k', linewidths=1).collections[0].set_label('area boundaries')
sv = clf.best_estimator_.support_vectors_
plt.scatter(sv[:, 0], sv[:, 1], facecolors='none', edgecolors='k',
            label='SV key points')
plt.legend()

#Plot source data with class labels
plt.subplot(1, 2, 2)
plt.title('Actual data')
plt.xlabel('PC1')
for target, color in zip(targets,colors):
    plt.scatter([pc[row][0] for row in range(len(y))
                 if y[row] == target],
                [pc[row][1] for row in range(len(y))
                 if y[row] == target], c = color, label=target)
plt.axis('off')
plt.legend()
plt.show()

# интересная задача: есть 4*10,5,2*2 и 1-рублевая монеты, надо найти вероятности
# для неполных наборов, сколько вариантов до 50 рублей они покроют

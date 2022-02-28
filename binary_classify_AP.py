import numpy as np
from sklearn import svm,datasets
import sklearn.metrics as metrics
from sklearn import model_selection
import matplotlib.pyplot as plt

iris=datasets.load_iris()
X=iris.data
y=iris.target

# add noisy features
random_state=np.random.RandomState(0)
n_samples,n_features=X.shape
X=np.c_[X,random_state.randn(n_samples,200*n_features)]

# split train and test
X_train,X_test,y_train,y_test=model_selection.train_test_split(X[y<2],y[y<2],
                                                               test_size=.5,
                                                               random_state=random_state)

# simple classifier
classifer=svm.LinearSVC(random_state=random_state)
classifer.fit(X_train,y_train)
y_score=classifer.decision_function(X_test)

# calculate average precision
average_precision=metrics.average_precision_score(y_test,y_score)

print("AP:{0:0.2f}".format(average_precision))

# plot precision-recall curve
disp=metrics.plot_precision_recall_curve(classifer,X_test,y_test)
disp.ax_.set_title("PR curve:AP={0:0.2f}".format(average_precision))
disp.plot()
plt.show()
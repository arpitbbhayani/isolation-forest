import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest


def do(random_state, datapoints, name="Isolation Forest"):
  # Generate train data
  X = datapoints
  # X_train = np.r_[X + 1.5, X - 1.5, X + 1.5, -1 * X - 1.5]
  X_train = X

  # Generate some abnormal novel observations
  X_test = random_state.uniform(low=-4, high=4, size=(100, 2))

  # fit the model
  clf = IsolationForest(max_samples=10, random_state=random_state, n_jobs=10, contamination=0.01)
  clf.fit(X_train)

  prediction = clf.predict(X_test)
  inliers = np.array([x for x, p in zip(X_test, prediction) if p == 1])
  outliers = np.array([x for x, p in zip(X_test, prediction) if p == -1])

  # plot the line, the samples, and the nearest vectors to the plane
  plt.title(name)

  b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white',
                  s=20, edgecolor='k')
  i = plt.scatter(inliers[:, 0], inliers[:, 1], c='green',
                  s=20, edgecolor='k')
  o = plt.scatter(outliers[:, 0], outliers[:, 1], c='red',
                  s=20, edgecolor='k')
  plt.axis('tight')
  plt.xlim((-5, 5))
  plt.ylim((-5, 5))
  # plt.legend([b1, i, o],
  #           ["training observations",
  #             "new regular observations", "new abnormal observations"],
  #           loc="upper left")
  plt.show()

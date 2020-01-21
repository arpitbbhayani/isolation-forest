import numpy as np
from isolation_forest import do


def plot_normal():
  random_state = np.random.RandomState(42)
  do(random_state, 0.3 * random_state.randn(1000, 2))

def plot_uniform():
  random_state = np.random.RandomState(42)
  do(random_state, 0.3 * random_state.uniform(low=-3, high=3, size=(1000, 2)))

def plot_beta():
  random_state = np.random.RandomState(42)
  do(random_state, 0.3 * random_state.beta(0, 2, size=(1000, 2)))

if __name__ == '__main__':
  plot_beta()

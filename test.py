import numpy as np


a = [np.clip(np.array([1,2,3]), -1, 2) for _ in range(3)]
print(a)
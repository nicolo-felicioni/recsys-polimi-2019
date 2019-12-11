import math

import numpy as np

def linear (mean = 0.5, offset = 0.5, size=30):
    min = mean - offset
    max = mean + offset
    if min < max:
        step = (max - min) / 10
        list = np.array([range(size)], dtype=np.float) + 1
        list = list * step + min
        list = list / list.max()
        delta = max - min
        list = list * delta + min
        return list.reshape(size)[::-1]
    else:
        return np.array([0] * size)

def exponential (mean = 0.5, offset = 0.5, size=30, base=1.5):
    min = mean - offset
    max = mean + offset
    if min < max:
        list = np.array([range(size)], dtype=np.float)
        list = base ** list
        list = list / list.max()
        delta = max - min
        list = list * delta + min
        return list.reshape(size)[::-1]
    else:
        return np.array([0] * size)

def logarithmic (mean = 0.5, offset = 0.5, size=30, base=1.5):
    min = mean - offset
    max = mean + offset
    if min < max:
        list = np.array([range(size)], dtype=np.float) + 1.1
        list = np.log(list) / np.log(base)
        list = list / list.max()
        delta = max - min
        list = list * delta + min
        return list.reshape(size)[::-1]
    else:
        return np.array([0] * size)

if __name__ == '__main__':
    print("hi")
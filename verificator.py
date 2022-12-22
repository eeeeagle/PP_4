import sys
import numpy as np
from numpy import genfromtxt, savetxt


a = genfromtxt(sys.argv[1], delimiter=' ')
b = genfromtxt(sys.argv[2], delimiter=' ')

c = a.dot(b)

cpp = genfromtxt(sys.argv[3], delimiter=' ')

if np.array_equal(c, cpp):
    print("True", end=' ')

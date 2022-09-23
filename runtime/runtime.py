#!/usr/bin/env python4

import numpy as np

SIZES = [4096, 8192, 16384, 32768]

class Ciphertext:
    def __init__(self, size=4096, arr=None):
        self.size = size
        if arr is not None:
            self.array = arr

        else:
            self.array = np.zeros(size)

    def load(self, x):
        n = len(x)
        assert(self.size >= n)
        self.array[:n] = x

    def get(self, n):
        return self.array[:n]

    def __add__(self, x):
        if isinstance(x, Ciphertext):
            return Ciphertext(self.size, self.array + x.array)

        else:
            return Ciphertext(self.size, self.array + x)

    def __sub__(self, x):
        if isinstance(x, Ciphertext):
            return Ciphertext(self.size, self.array - x.array)

        else:
            return Ciphertext(self.size, self.array - x)
        
    def __mul__(self, x):
        if isinstance(x, Ciphertext):
            return Ciphertext(self.size, self.array * x.array)

        else:
            return Ciphertext(self.size, self.array * x)

    def rotate(self, n):
        return Ciphertext(self.size, self.array[[(i-n) % self.size for i in range(self.size)]])

    def __str__(self):
        return str(self.array)

# transformations done at client-side:
# - pad         (use np.pad)
# - project     (use indexing)
# - transpose   (use np.transpose)
# - fill        (use np.stack, nested)
# - stride (?)
#
# use np.reshape(-1) to shape into 1-dim vector for encoding into ciphertext
# 

# example: 2x2 matrix-matrix multiply
# code:
#
#   for i in (0, 2) {
#       for j in (0, 2) {
#           sum(for k in (0, 2) { A[i][k] * B[k][j] })
#       }
#   }
#   - you shouldn't need to define index extents like this explicitly;
#     this can be done using interval analysis
#
# dimension order: k j i
# after filling, A has dimensions i k j, so transpose to (1 2 0)
# after filling, B has dimensions k j i, so transpose to (0 1 2)
# index-free normal form: sum(transpose(fill(A, 2), (1, 0, 2)) * transpose(fill(B, 2), (0, 1, 2)), axis=0)

# x:    array to fill
# dims: list of sizes for each new dimension
def fill(x, dims):
    cur = x
    for d in dims:
        cur = np.stack([cur]*d)

    return cur

def transpose(x, pi):
    # transpose() reverses axes, which occurs bc np.stack
    # adds dims to the FRONT of the array
    return np.transpose(np.transpose(x), pi)


if __name__ == "__main__":
    a = np.array([1,2,3,4]).reshape((2,2))
    b = np.array([5,6,7,8]).reshape((2,2))

    print("input A:")
    print(a)
    print("input B:")
    print(b)

    # client preprocessing
    
    ta = transpose(fill(a, [2]), (1,2,0))
    tb = transpose(fill(b, [2]), (0,1,2))

    ca = Ciphertext(16)
    ca.load(ta.reshape(-1))

    cb = Ciphertext(16)
    cb.load(tb.reshape(-1))

    # server-side compute

    cprod1 = ca * cb
    cprod2 = cprod1.rotate(-4)
    csum = cprod1 + cprod2

    # client post-processing

    out = csum.get(4).reshape((2,2))
    print("result:")
    print(out)



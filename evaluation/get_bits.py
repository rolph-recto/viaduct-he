import itertools
import sys

if len(sys.argv) < 2:
    print("usage: get_bits.py [num bits]")

n = int(sys.argv[1])
bits = [[0,1] for _ in range(n)]

for bitvec in itertools.product(*bits):
    print(list(bitvec))

# Benchmarks 

## Research Question 1
RQ1: Can the compiler generate efficient programs with complex layouts?

Compare with a "semi-naive" baseline that tries to vectorize;
this is "semi-naive" because the "naive" baseline is to *not*
vectorize at all and use one slot per ciphertext

## Benchmarks
1. private information retrieval (retrieval-256)
vec-size = 4096
- all input arrays become single vectors

baseline: 1 key per vector

2. PIR with bigger database (retrieval-1024)
vec-size = 4096
- shard keys into 2 vectors

3. private set union (set-union)
vec-size = 8192

baseline: 1 key per vector
change DB size to 16 to fit 8192

4. double matmul (matmul-2) (from CHET)
vec-size = 4096

e1-o0
val v_B_1: C = vector(B[0, 0]<{1:16::1}, {0:16::1}, {16}>)
val v_A1_1: N = vector(A1[0, 0]<{16}, {1:16::1}, {0:16::1}>)
val v_A2_1: N = vector(A2[0, 0]<{16}, {0:16::1}, {1:16::1}>)

e2-o0
val v_B_1: C = vector(B[0, 0]<{0:16::1}, {16}, {1:16::1}>)
val v_A1_1: N = vector(A1[0, 0]<{1:16::1}, {0:16::1}, {16}>)
val v_A2_1: N = vector(A2[0, 0]<{0:16::1}, {1:16::1}, {16}>)

5. hamming distance (distance)
compare a user test point with a set of test points
vec-size: 2048
* want to show off diagonal layout

- infeasible on 4096

baseline: one point per vector (row-wise layout)

6. single-input single-output convolution (conv-siso) (from gazelle)
vec-size: 4096

baseline: sliding windows (i.e. store input pixels that will contribute to a
single output pixel)

7. single-input multiple-output convolution (conv-simo) (from gazelle)
vec-size: 4096

baseline: sliding windows

## Research Question 2
RQ2: Can the compiler optimize circuits efficiently?

### Benchmarks
- use Porcupine benchmarks ?
- fix layout

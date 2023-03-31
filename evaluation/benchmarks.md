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
- shard keys into 5 vectors

3. private set union (set-union)
vec-size = 16384

baseline: 1 key per vector

4. double matmul (2-matmul)
vec-size = 4096

5. hamming distance (hamming-distance)
compare a user test point with a set of test points
vec-size: 2048
* want to show off diagonal layout

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

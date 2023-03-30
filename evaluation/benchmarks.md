# Benchmarks 

## Research Question 1
RQ1: Can the compiler generate efficient programs with complex layouts?

Compare with a "semi-naive" baseline that tries to vectorize;
this is "semi-naive" because the "naive" baseline is to *not*
vectorize at all and use one slot per ciphertext

## Benchmarks
1. private information retrieval (retrieval)
vec-size = 8192
- do another one that showcases tiling by doubling database size

2. private set union (set-union)
vec-size = 16384

3. double matmul (matmul2)
vec-size = 4096

4. L2 distance (l2-distance)
compare a user test point with a set of test points
vec-size: 2048
* want to show off diagonal layout

5. single-channel convolution (from gazelle)

6. multiple-channel convolution (from gazelle)

## Research Question 2
RQ2: Can the compiler optimize circuits efficiently?

### Benchmarks
- use Porcupine benchmarks
- fix layout

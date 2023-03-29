# Benchmarks 

## Research Question 1
RQ1: Can the compiler generate efficient programs with complex layouts?

## Benchmarks
1. private information retrieval (retrieval)
vec-size = 8192

2. private set union (set-union)
vec-size = 16384

3. double matmul (matmul2)
vec-size = 4096

4. L2 distance (l2-distance)
compare a user test point with a set of test points
vec-size: 2048?
* want to show off diagonal layout

5. mat-vec
vec-size: 2048 (?)
* show off tiling + diagonal layout (hybrid layout from gazelle)

## Research Question 2
RQ2: Can the compiler optimize circuits efficiently?

### Benchmarks
- use Porcupine benchmarks
- fix layout

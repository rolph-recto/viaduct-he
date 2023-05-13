# Viaduct-HE

Viaduct-HE is a compiler from array programs to vectorized homomorphic encryption.
It is designed primarily to generate HE programs that efficiently use the
SIMD capabilities (also known as "batching") of HE schemes such as BFV, BGV, and
CKKS. Batching allows HE programs to be executed with much fewer operations,
which depending on the program can result in orders of magnitude in speedup.

## Why Another HE Compiler?

There are a lot of HE compilers out there; see
[this SoK paper](https://arxiv.org/pdf/2101.07078.pdf) for an extensive list.
Most of these compilers do not automatically generate vectorized HE programs;
automatic vectorization has only been recently been tackled by compilers
such as
[Porcupine](https://dl.acm.org/doi/abs/10.1145/3453483.3454050),
[HECO](https://arxiv.org/pdf/2202.01649.pdf),
and [Coyote](https://dl.acm.org/doi/10.1145/3582016.3582057).

Viaduct-HE is distinguished among vectorizing HE compilers by having an
*array-oriented* source language. This allows the compiler to have a simple
representation of possible *vectorization schedules*, which represent how
data is laid out in ciphertexts. Having a simple representation for
vectorization schedules allows the compiler to easily search for efficient
schedules.

The idea of using an array-oriented source language to allow the straightforward
representation of vectorization schedules is directly inspired by the
[Halide](https://halide-lang.org/) compiler for image processing pipelines.

Viaduct-HE is an offshoot of the [Viaduct](github.com/apl-cornell/viaduct) compiler.

## Example

Consider the following Viaduct-HE program that multiples three 16x16 matrices
together.

```
input A1: [16,16] from client
input A2: [16,16] from client
input B: [16,16] from client
let res =
    for i: 16 {
        for j: 16 {
            sum(for k: 16 { A1[i][k] * B[k][j] })
        }
    }
in
for i: 16 {
    for j: 16 {
        sum(for k: 16 { A2[i][k] * res[k][j] })
    }
}
```

Note that this is quite close to what you would write in a traditional
imperative language:

```
A1  = Array(16,16);
A2  = Array(16,16);
B   = Array(16,16);

res = Array(16,16);
for (i = 0; i < 16; i++) {
    for (j = 0; j < 16; j++) {
        acc = 0;
        for (k = 0; k < 16; k++) {
            acc += A1[i][k] * B[k][j];
        }
        res[i][j] = acc;
    }
}

out = Array(16,16);
for (i = 0; i < 16; i++) {
    for (j = 0; j < 16; j++) {
        acc = 0;
        for (k = 0; k < 16; k++) {
            acc += A2[i][k] * res[k][j];
        }
        out[i][j] += acc;
    }
}
```

The compiler finds an efficient schedule that lays out each of the three input
matrices in a single ciphertext. The schedule defines a different
data layout for each ciphertext; this lines up the elements just so, such that
all 4096 (16^3) of the multiplication operations for a matrix multiplication can
be done *simultaneously* in a single homomorphic multiplication operation! 
The result of the computation is stored in the `v_out` ciphertext, which is
sent back to the client.

```
// the vector() operation defines directives to pack array elements into a ciphertext
val v_A2 = vector(A2[0, 0]<{0:4::1}, {4}, {1:4::1}>)
val v_A1 = vector(A1[0, 0]<{1:4::1}, {4}, {0:4::1}>)
val v_B = vector(B[0, 0]<{0:4::1}, {1:4::1}, {4}>)

// these define operations over ciphertexts;
// + and * are homomorphic addition and multiplication respectively, while
// rot is rotation, which cyclically shifts data elements in a ciphertext
var v_res
i2 = v_A1 * v_B
i3 = rot -32 i2
i4 = i2 + i3
i5 = rot -16 i4
i6 = i4 + i5
res = i6

var v_out
i9 = v_A2 * res
i10 = rot -2 i9
i11 = i9 + i10
i12 = rot -1 i11
i13 = i11 + i12
v_out = i13
```

## Installation

NOTE: The installation and building process below has only been tested on
Apple Silicon Macs (i.e. M1/M2). Your process might vary!

The compiler is written in Rust. To install it, clone this repository and
run `cargo build`. To install, run: `cargo install --path .`

By default, the compiler is configured to use the LP extractor of the
`egg` equality saturation library, which requires the CBC COIN-OR library
(`libcbc`).

You can install the library in your system using the appropriate command below:

| Operating System  | Command |
|-------------------|---------|
| Fedora / Red Hat	| dnf install coin-or-Cbc-devel |
| Ubuntu / Debian   | apt-get install coinor-libcbc-dev |
| macOS	            | brew install cbc |

You also have to ensure that the link path contains `libcbc`.
The `build.rs` file currently adds a `cargo:rustc-link-search` directive 
to add `/opt/homebrew/lib` to the linker's search path, which is the default
Homebrew installation directory in Apple Silicon Macs. If your system is
different, you might have to add to the linker search path as well, depending on
where your package manager installs `libcbc`.

## Compiling a Program

You can compile a Viaduct-HE program as follows:

```
viaducthe [SOURCE FILE] -o [OUTPUT FILE]
```

There are a lot more options and flags you can pass to the compiler to tune
the compilation process; run `viaducthe --help` to get a list.

## Executing HE Programs

The compiler currently supports a back end for the
[PySEAL](https://github.com/Lab41/PySEAL) library, which provides Python
bindings for the [Microsoft SEAL](https://github.com/microsoft/SEAL)
homomorphic encryption library.

The PySEAL backend generates a Python file that you can execute as a regular
script. You need to pass the `-t [TEMPLATE FILE]` flag to specify the template
file used during code generation. The default template file for the PySEAL
backend is `pyseal_template.txt` in this directory.

To run the Python 3 file generated by the Viaduct-HE's PySEAL backend,
you need to copy the shared library (`pyseal*.so` or `pyseal*.dll`) generated
by PySEAL to the same directory as the generated file. You also need the
[numpy](https://numpy.org/) library installed in your Python environment.
Once these dependencies are satisfied, you can then run
`python3 generated_file.py` to execute the HE program.

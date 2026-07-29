# Fast Einsum with Batch Matrix Multiplication
This repository contains an einsum python library that uses batch matrix multiplication created by
- Sonja Weitzing and
- Erik Henicke


in the context of the course Algorithm Engineering by Mark Blacher at Friedrich Schiller University Jena. We compared the library and described the results in the [project paper](paper.pdf).

## Motivation
Einsum is a versatile tool for many multi-linear tensor operations, made popular by NumPy's implementation in 2011. It has great expressive power and is part of major machine learning frameworks such as PyTorch and TensorFlow, leading to its widespread use in deep learning. Since large parts of deep learning are series of matrix multiplications and tensor contractions, they can be easily mapped to einsum expressions.

The approach of reducing tensor contractions to batch matrix multiplication (BMM) is known as [Transpose-Transpose-GEMM-Transpose (TTGT)](http://publications.rwth-aachen.de/record/755345/files/755345.pdf), with the only difference that batch dimensions may be present. The input tensors must be transposed and reshaped so that they can be interpreted as batches of matrices. This allows us to generalize over all contracting einsum expressions. The translation between tensors and matrices relies on the [einsum_bmm](https://github.com/jcmgray/einsum_bmm/blob/main/einsum_bmm.py) approach by Johnnie Gray.

Our main work focuses on implementing a fast BMM in C++ using various optimization techniques — kernelization with AVX2 vector intrinsics, cache-friendly blocking, and OpenMP parallelization — and evaluating it in a comprehensive benchmark. Our optimized implementations outperform NumPy's einsum function by up to 100x for large input shapes.

![Einsum Benchmark](images/einsum_custom_plot_by_shapes.png)
*Performance comparison of einsum implementations. The benchmark shows significant performance gains of the BMM approach over NumPy's einsum, especially for large input shapes. Benchmark instances are represented by their shapes when interpreted as matrices.*

## Folder structure
- `bmm` contains the `C++` library for batch matrix multiplication (`bmm/src`), with its tests in `bmm/tests` (C++/Catch2).
- `fast_einsum` contains the python library that uses the `C++` library for batch matrix multiplication (`fast_einsum/src`), with its tests in `fast_einsum/tests` (Python/pytest).
- `einsum_benchmark` contains the benchmark and tests for the fast einsum library.
- `results` contains the results of the benchmarks.
- `plot` contains the scripts to plot the results.

## Installation
`fast_einsum` ships as source: its C++ extension (built from the `bmm` library) is compiled automatically when the
package is installed, so there is no prebuilt wheel to download.

### Requirements
- Python >= 3.10
- A C++ compiler with OpenMP support, and CMake >= 3.15
- A BLAS implementation with development headers (e.g. `libopenblas-dev` on Debian/Ubuntu)
- An x86_64 CPU with AVX2 support — the BMM kernels use AVX2 intrinsics and are compiled with `-march=native`

### With uv (recommended)
```bash
uv sync --extra test
```
This resolves dependencies, compiles the C++ extension via CMake, and installs everything into a local `.venv`.
Drop `--extra test` if you don't need the test dependencies.

### With pip
```bash
pip install ".[test]"
```

## Running tests
```bash
uv run pytest fast_einsum/tests -v
```
or, without uv, in your activated environment:
```bash
pytest fast_einsum/tests -v
```

## Support
If any support is needed, we are there to help. Reach out to us under
- erik.henicke@uni-jena.de or
- sonja.marina.weitzing@uni-jena.de

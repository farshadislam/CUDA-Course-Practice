# Faster Matrix Operations using GPU Multithreading

Goal here was pretty straightforward: implement a functions that are able to use the parallel programming capa\biltities of the NVIDIA Gpu in my computer to perform matrix addition and multiplication far more efficiently than would be possible only using the CPU.

These computations are especially popular in both the machine learning and crypto mining spaces for the simple reason that they're fast -- VERY fast. Admittedly, there are far more advanced implementations utilizing this technology besides matrix operations, such as real-time image/video processing, data analytics, cryptography, deep learning among many other use-cases.

Regardless, both files in this repository display the immediate advantages of CUDA programming, and will be explained in-depth.

---

## `matrix_addition_2d.cu`

This is about as simple as implementing functions with CUDA can get. Typically, using a CPU, every element in two matrices being added together would have to be computed sequentially -- one after the other -- before a valid solution could be found. However, using the cores in my NVIDIA GPU, these additions can actually occur AT THE SAME TIME. This is essense of multithreading: take simple operations to be applied to a dataset, then compute all of them parallel to one another in order to save on runtime.

![GPU matrix addition occurs 32x faster than standard CPU computation](./Screenshot%202025-09-08%20192236.png)

---

## `matrix_multiplication.cu`

Same idea here but this also introduces the idea of tiled matrix multiplication using shared memory (wowwwww). This is supposed to be even more efficient than standard mutlithreading due to the fact that all the computations happen within the same thread block. And sure enough, it actually does provide a significant improvement upon the standard GPU method.

![Benchmarks for different ways of multiplaying matrices shown sequentially](./Screenshot%202025-09-07%20181458.png)
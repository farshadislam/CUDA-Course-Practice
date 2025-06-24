#include <stdio.h>        // Standard input output (printf, scanf)
#include <stdlib.h>       // Standard library helper (free, exit, malloc)
#include <time.h>         // Used to analyze runtime of functions
#include <cuda_runtime.h> // Wraps CUDA API routines
#include <math.h>         // Math
#include <iostream>       // Allows for use of C++ standard input/output

#define N 8 // numRows in matrix
#define M 4 // numCols in matrix
#define BLOCK_SIZE_X 8
#define BLOCK_SIZE_Y 4

void cpu_vector_addition(int *a, int *b, int *c, int numRows, int numCols)
{
    for (int i = 0; i < numRows; i++)
    {
        for (int j = 0; j < numCols; j++)
        {
            // c[i][j] = a[i][j] + b[i][j]; --> (Would be written like this if I were using double pointers)
            int trueIndex = i * numCols + j; // Instead I appeal to row-major order
            c[trueIndex] = a[trueIndex] + b[trueIndex];
        }
    }
}

void gpu_vector_addition(int *a, int *b, int *c, int numRows, int numCols)
{
    int i = blockIdx.x       // Block index inside of the grid
                * blockDim.x // Number of threads per block
            + threadIdx.x;   // Thread inside of the block

    int j = blockIdx.y * blockDim.y + threadIdx.y; // Same story for y, except along the second axis

    if (i < numRows && j < numCols)
    {
        int trueIndex = i + j * numRows; // Row major order actually happens in the OPPOSITE specificity compared to Assembly
        if (trueIndex < numRows * numCols)
        {
            c[trueIndex] = a[trueIndex] + b[trueIndex];
        }
    }
}

// Copied directly from CUDA course: https://github.com/Infatoshi/cuda-course/blob/master/05_Writing_your_First_Kernels/02%20Kernels/00_vector_add_v1.cu
void init_vector(int *vec, int numRows, int numCols)
{
    for (int i = 0; i < numRows; i++)
        for (int j = 0; j < numCols; j++)
        {
            int trueIndex = i + j * numRows;
            vec[trueIndex] = rand() % 10; // Returns a pseudo-random integer between 0 and 9
        }
}

// Copied directly from CUDA course: https://github.com/Infatoshi/cuda-course/blob/master/05_Writing_your_First_Kernels/02%20Kernels/00_vector_add_v1.cu
double get_time()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main()
{
    /* Pointer initialization */
    int *h_a, *h_b, *h_c_cpu, *h_c_gpu; // Host variables (standard operations)
    int *d_a, *d_b, *d_c_gpu;           // Device variables (CUDA operations)
    size_t size = N * M * sizeof(int);  // size_t is an unsigned long integer that gives more exact bit precision when necessary

    /* Allocating memory to host variables */
    h_a = (int *)malloc(size);
    h_b = (int *)malloc(size);
    h_c_cpu = (int *)malloc(size);
    h_c_gpu = (int *)malloc(size);

    /* Give vectors their random integers */
    srand(time(NULL)); // Pseudo-random number generator is seeded according to PC's internal clock (woah)
    init_vector(h_a, N, M);
    init_vector(h_b, N, M);

    /* Allocating memory for device */
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c_gpu, size);

    /* Copy common data from host to device */
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    /* Grid and block dimensions in 2D */
    dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridDim(
        (N + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
        (M + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);

    /* Warm up runs */
    printf("Warm up runs:\n");
    for (int i = 0; i < 3; i++)
    {
        cpu_vector_addition(h_a, h_b, h_c_cpu, N, M);
        gpu_vector_addition<<<gridDim, blockDim>>>(d_a, d_b, d_c_gpu, N, M);
        cudaDeviceSynchronize();
    }

    /* Runtime in milliseconds for CPU implementation */
    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 5; i++)
    {
        double start_time = get_time();
        cpu_vector_addition(h_a, h_b, h_c_cpu, N, M);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 5.0;

    /* Runtime in milliseconds for GPU implementation */
    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 100; i++)
    {
        cudaMemset(d_c_gpu, 0, size); // Clear previous results
        double start_time = get_time();
        gpu_vector_addition<<<gridDim, blockDim>>>(d_a, d_b, d_c_gpu, N, M);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 100.0;

    /* Verify that both outputs are the same */
    cudaMemcpy(h_c_gpu, d_c_gpu, size, cudaMemcpyDeviceToHost);
    bool correct_1d = true;
    for (int i = 0; i < N; i++)
    {
        if (fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-4)
        {
            correct_1d = false;
            std::cout << i << " cpu: " << h_c_cpu[i] << " != " << h_c_gpu[i] << std::endl;
            break;
        }
    }
    printf("1D Results are %s\n", correct_1d ? "correct" : "incorrect");

    printf("CPU average time: %f milliseconds\n", cpu_avg_time * 1000);
    printf("GPU 1D average time: %f milliseconds\n", gpu_avg_time * 1000);
    printf("Speedup (CPU vs GPU 1D): %fx\n", cpu_avg_time / gpu_avg_time);

    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    free(d_a);
    free(d_b);
    free(d_c_gpu);

    return 0;
}
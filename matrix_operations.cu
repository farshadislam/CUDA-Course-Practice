#include <stdio.h>        // Standard input output (printf, scanf)
#include <stdlib.h>       // Standard library helper (free, exit, malloc)
#include <time.h>         // Used to analyze runtime of functions
#include <cuda_runtime.h> // Wraps CUDA API routines

#define N 8 // numRows in matrix
#define M 4 // numCols in matrix

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
    int *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    int *d_a, *d_b, *d_c_gpu;
    size_t size = N * M * sizeof(int); // size_t is an unsigned long integer that gives more exact bit precision when necessary

    /* Allocating memory to host variables */
    h_a = (int *)malloc(size);
    h_b = (int *)malloc(size);
    h_c_cpu = (int *)malloc(size);
    h_c_gpu = (int *)malloc(size);

    /* Give vectors their random integers*/
    srand(time(NULL)); // Pseudo-random number generator is seeded according to PC's internal clock (woah)
    init_vector(h_a, N, M);
    init_vector(h_b, N, M);

    /* Allocating memory for device*/
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c_gpu, size);

    /* Copy common data from host to device */
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    return 0;
}
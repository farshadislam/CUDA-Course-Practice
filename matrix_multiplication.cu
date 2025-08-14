#include <stdio.h>        // Standard input output (printf, scanf)
#include <stdlib.h>       // Standard library helper (free, exit, malloc)
#include <time.h>         // Used to analyze runtime of functions
#include <cuda_runtime.h> // Wraps CUDA API routines
#include <math.h>         // Math
#include <iostream>       // Allows for use of C++ standard input/output

#define TILE_SIZE 16

void matmul(int *matrixA, int *matrixB, int *matrixC, int m, int n, int k)
{
    for (int i = 0; i < m; i++) // For rows of A
    {
        for (int j = 0; j < n; j++) // For columns of B
        {
            int valueC = 0;             // What the product that goes into position C[x][y] will be
            for (int l = 0; l < k; l++) // Means of iterating through non-matching matrices for products
            {
                valueC += matrixA[i * k + l]    // i * k always goes to first position in row, l iterates through what remains
                          * matrixB[j + n * l]; // j always iterates through column specified by n * l
            }
            matrixC[i * n + j] = valueC; // Product is passed into C[x][y] until full
        }
    }
}

/* Basic matrix multiplication using GPU threads*/
__global__ void gpu_matmul(int *matrixA, int *matrixB, int *matrixC, int m, int n, int k)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Block index in grid * # of threads per block * thread index inside of block
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Similar form: i * cols + j

    if (i < m && j < n)
    { // Iterators bound by the length and width of the 2D matrix
        int coordValue = 0;
        for (int l = 0; l < k; l++)
        { // Limited iterator that only loops through k operations
            coordValue += matrixA[i * k + l] + matrixB[j + n * l];
        }
        matrixC[i * n + j] = coordValue;
    }
}

/// @brief Supposedly even more efficient than GPU matmul, due to its use of shared memory
/// @param matrixA
/// @param matrixB
/// @param matrixC
/// @param m
/// @param n
/// @param k
/// @return N/A

__global__ void tiled_gpu_matmul(int *matrixA, int *matrixB, int *matrixC, int m, int n, int k)
{
    __shared__ int sharedA[TILE_SIZE][TILE_SIZE]; // Tile size is large enough to accomodate matrices with dimensions of at least 1024 along either axes
    __shared__ int sharedB[TILE_SIZE][TILE_SIZE]; // On-chip, fast memory

    int bx = blockIdx.x, by = blockIdx.y;   // Track block position
    int tx = threadIdx.x, ty = threadIdx.y; // Track thread position

    int row = by * TILE_SIZE + ty; // Actual coordinates of thread by array location in memory
    int col = bx * TILE_SIZE + tx;

    int sum = 0; // Dot product for (row, col)

    for (int tile = 0; tile < (k + TILE_SIZE - 1) / TILE_SIZE; ++tile)
    {                                                                   // Covering all tiles with ceiling division, along shared axis k
        if (row < m && tile * TILE_SIZE + tx < k)                       // Keep within matrix edges
            sharedA[ty][tx] = matrixA[row * k + tile * TILE_SIZE + tx]; // Load element from A into shared memory
        else
            sharedA[ty][tx] = 0; // Boundary check ensures that anything out of matrix bounds goes to zero

        if (col < n && tile * TILE_SIZE + ty < k) // Same dealio here as lines 68-71
            sharedB[ty][tx] = matrixB[(tile * TILE_SIZE + ty) * n + col];
        else
            sharedB[ty][tx] = 0.0f;

        __syncthreads(); // Ensures that all tiles have loaded before starting computations

        for (int i = 0; i < TILE_SIZE; ++i)
            sum += sharedA[ty][i] * sharedB[i][tx]; // Partial dot product computed using tile

        __syncthreads(); // Everything syncs again so that all threads are finished being used before new shared memory accessing
    }

    if (row < m && col < n)
    {
        matrixC[row * n + col] = sum; // Each thread writes its accumulated sum into the correct coordinate of the C matrix
    }

    // Beautiful explanation of this procedure here: https://penny-xu.github.io/blog/tiled-matrix-multiplication
}

void init_matrix(int *whateverMatrix, int rows, int cols)
{
    for (int i = 0; i < rows; i++) // For rows of A
    {
        for (int j = 0; j < cols; j++) // For columns of B
        {
            whateverMatrix[i * cols + j] = rand() / RAND_MAX;
            // Needs to index through array like this because of the pointer array
        }
    }
}

/// @brief Just gets the amount of time that occurs between operations
/// @return The time in seconds, with many, MANY trailing decimal points
double get_time()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main()
{
    const int M = 1024; // Unchanging integers
    const int N = 1024;
    const int K = 1024;

    int *h_A, *h_B, *h_C_cpu, *h_C_gpu;
    int *d_A, *d_B, *d_C;

    // Calculate matrix sizes in bytes
    size_t size_A = M * K * sizeof(int);
    size_t size_B = K * N * sizeof(int);
    size_t size_C = M * N * sizeof(int);

    // Declare device pointers
    float *d_A, *d_B, *d_C;

    // Allocate device memory
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // Kernel launch code
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    tiled_gpu_matmul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

    // Synchronize device
    cudaDeviceSynchronize();

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Check for any CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    return 0;
}
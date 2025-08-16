#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <iostream>

#define TILE_SIZE 16
#define BLOCK_SIZE 32

// CPU matrix multiplication
void matmul_cpu(int *A, int *B, int *C, int M, int N, int K)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int sum = 0;
            for (int l = 0; l < K; l++)
                sum += A[i * K + l] * B[l * N + j]; // M * K (matrix A) times K * N (matrix B) gives M * N (matrix C)
            C[i * N + j] = sum;
        }
    }
}

// GPU naive matrix multiplication
__global__ void matmul_gpu(int *A, int *B, int *C, int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) // Bound check
    {
        int sum = 0;
        for (int l = 0; l < K; l++)
            sum += A[row * K + l] * B[l * N + col]; // Multiple threads are doing this operation on different 2D array elements at the same time
        C[row * N + col] = sum;
    }
}

// GPU tiled matrix multiplication using shared memory
__global__ void matmul_tiled(int *A, int *B, int *C, int M, int N, int K)
{
    __shared__ int tileA[TILE_SIZE][TILE_SIZE];
    __shared__ int tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    int sum = 0;

    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++)
    {
        int aCol = tile * TILE_SIZE + threadIdx.x;
        int bRow = tile * TILE_SIZE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0; // Map element of A to tile
        tileB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0; // Map element of B to tile

        __syncthreads(); // Wait until every element has been loaded into tiles

        for (int i = 0; i < TILE_SIZE; i++)
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x]; // Calculate dot product of elements in tiles and get full product matrix C

        __syncthreads(); // Wait for all threads to finish performing this operation before moving to the next step
    }

    if (row < M && col < N)
        C[row * N + col] = sum; // Fill every element in matrix C
}

// Initialize matrix with random ints
void init_matrix(int *matrix, int rows, int cols)
{
    for (int i = 0; i < rows * cols; i++)
        matrix[i] = static_cast<int>(rand()) / RAND_MAX * 10; // values 0..10
}

// Timing helper
double get_time()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main()
{
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    int *h_A, *h_B, *h_C_cpu, *h_C_gpu;
    int *d_A, *d_B, *d_C;
    int *t_A, *t_B, *t_C;

    size_t size_A = M * K * sizeof(int);
    size_t size_B = K * N * sizeof(int);
    size_t size_C = M * N * sizeof(int);

    h_A = (int *)malloc(size_A);
    h_B = (int *)malloc(size_B);
    h_C_cpu = (int *)malloc(size_C);
    h_C_gpu = (int *)malloc(size_C);

    srand(time(NULL));
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Warm-up
    for (int i = 0; i < 5; i++) {
        matmul_cpu(h_A, h_B, h_C_cpu, M, N, K);
        matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
    }
    
    // CPU timing
    double cpu_time = 0;
    for (int i = 0; i < 5; i++)
    {
        double start = get_time();
        matmul_cpu(h_A, h_B, h_C_cpu, M, N, K);
        cpu_time += get_time() - start;
    }
    cpu_time /= 5;

    // GPU timing
    double gpu_time = 0;
    for (int i = 0; i < 5; i++)
    {
        double start = get_time();
        matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
        gpu_time += get_time() - start;
    }
    gpu_time /= 5; // Get average of five runs

    // Copy GPU result back
    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);

    printf("CPU time: %f ms\n", cpu_time * 1000);
    printf("GPU time: %f ms\n", gpu_time * 1000);
    printf("Speedup from CPU time by %fx", cpu_time / gpu_time);

    bool correct = true;
    for (int i = 0; i < M * N; i++)
    {
        if (fabs(h_C_cpu[i] - h_C_gpu[i]) > 1e-3)
        {
            correct = false;
            break;
        }
    }
    printf("Results match: %s\n", correct ? "YES" : "NO");

    /* Now to demonstrate the power of tiled matmul! */

    cudaMalloc(&t_A, size_A);
    cudaMalloc(&t_B, size_B);
    cudaMalloc(&t_C, size_C);

    cudaMemcpy(t_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(t_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Kernel launch code
    dim3 blockDim2(TILE_SIZE, TILE_SIZE);
    dim3 gridDim2((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_tiled<<<gridDim2, blockDim2>>>(d_A, d_B, d_C, M, N, K);

    // Warm-up
    for (int i = 0; i < 5; i++) {
        matmul_cpu(h_A, h_B, h_C_cpu, M, N, K);
        matmul_tiled<<<gridDim2, blockDim2>>>(t_A, t_B, t_C, M, N, K);
        cudaDeviceSynchronize();
    }

    double gpu_tiled_time = 0;
    for (int i = 0; i < 5; i++)
    {
        double start = get_time();
        matmul_tiled<<<gridDim2, blockDim2>>>(t_A, t_B, t_C, M, N, K);
        cudaDeviceSynchronize();
        gpu_time += get_time() - start;
    }
    gpu_tiled_time /= 5; // Get average of five runs

    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);

    printf("\n\nCPU time: %f ms\n", cpu_time * 1000);
    printf("Tiled GPU time: %f ms\n", gpu_time * 1000);
    printf("Speedup from CPU time by %fx", cpu_time / gpu_time);

    correct = true;
    for (int i = 0; i < M * N; i++)
    {
        if (fabs(h_C_cpu[i] - h_C_gpu[i]) > 1e-3)
        {
            correct = false;
            break;
        }
    }
    printf("Results match: %s\n", correct ? "YES" : "NO");

    free(h_C_gpu); // Allow for the memory here to be accessed by tiled matmul

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    
    cudaFree(t_A);
    cudaFree(t_B);
    cudaFree(t_C);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    return 0;
}

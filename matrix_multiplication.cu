#include <stdio.h>        // Standard input output (printf, scanf)
#include <stdlib.h>       // Standard library helper (free, exit, malloc)
#include <time.h>         // Used to analyze runtime of functions
#include <cuda_runtime.h> // Wraps CUDA API routines
#include <math.h>         // Math
#include <iostream>       // Allows for use of C++ standard input/output

#define M 150
#define N 300

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

__global__ void gpu_vector_addition(int *matrixA, int *matrixB, int *matrixC, int m, int n, int k)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Block index in grid * # of threads per block * thread index inside of block
    int j = blockIdx.x * blockDim.x + threadIdx.x;

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
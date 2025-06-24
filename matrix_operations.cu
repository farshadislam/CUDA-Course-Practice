#include <stdio.h>        // Standard input output (printf, scanf)
#include <stdlib.h>       // Standard library helper (free, exit, malloc)
#include <time.h>         // Used to analyze runtime of functions
#include <cuda_runtime.h> // Wraps CUDA API routines

#define N = 8 // numRows in matrix
#define M = 4 // numCols in matrix

void cpu_vector_addition(int *a, int *b, int *c, int numRows, int numCols)
{
    for (int i = 0; i < numRows; i++)
    {
        for (int j = 0; j < numCols; j++)
        {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
}

void gpu_vector_addition(int *a, int *b, int *c, int numRows, int numCols)
{
    int i = blockIdx.x                             // Block index inside of the grid
                * blockDim.x                       // Number of threads per block
            + threadIdx.x;                         // Thread inside of the block
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Same story for y, except along the second axis

    if (i < numRows && j < numCols)
    {
        int idx = i + j * numRows
    }
}
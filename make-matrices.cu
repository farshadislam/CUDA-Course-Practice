#include <iostream>
#include <fstream>
#include <vector>
#include <random>
using namespace std;

#define main_x 10
#define main_y 100

 void make_matrix(int numRows, int numCols, float *storeHere) {
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            int elemAccess = i * numCols + j;

            std::random_device rd;  
            std::mt19937 gen(rd());  // Mersenne Twister RNG
            std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
            
            storeHere[elemAccess] = dist(gen);
        }
    }
 };

 void write_matrix(ofstream &file, float *storeHere, int numRows, int numCols) {
    for (size_t i = 0; i < numRows; i++) {
        for (size_t j = 0; j < numCols; j++) {
            file << storeHere[i * numCols + j];
            if (j != numCols - 1) {
                file << ",";
            }
        }
        file << "\n";
    }   
}

 int main() {
    float *matrixA, *matrixB;
    size_t matrix_size = main_x * main_y * sizeof(float);

    matrixA = (float *)malloc(matrix_size);
    matrixB = (float *)malloc(matrix_size);

    make_matrix(main_x, main_y, matrixA);
    make_matrix(main_x, main_y, matrixB);

    ofstream output("big-matrices.txt");
    if (!output.is_open()) {
        cerr << "Error: Could not open file for writing.\n";
        return 1;
    }

    write_matrix(output, matrixA, main_x, main_y);
    output << "\n";
    write_matrix(output, matrixB, main_x, main_y);

    output.close();
    
    return 0;
 }
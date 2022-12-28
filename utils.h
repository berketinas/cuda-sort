#ifndef UTILS_H
#define UTILS_H

void handleError(cudaError_t code);
void init(int* array, int length);
void printArray(int* array, int length);
void checkSorted(int* array, int length);

#endif
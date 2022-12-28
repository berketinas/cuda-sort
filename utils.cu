#include "includes.h"

void handleError(cudaError_t code) {
    if(code != cudaSuccess) {
        fprintf(stderr, "GPU error: %s\n", cudaGetErrorString(code));
        if(abort) exit(code);
    }
}

void init(int* array, int length) {
    for(int i = 0; i < length; i++) array[i] = rand();
}

void printArray(int* array, int length) {
    printf("Array: [");
    for(int i = 0; i < length - 1; i++) printf("%d, ", array[i]);
    printf("%d]\n", array[length - 1]);
}

void checkSorted(int* array, int length) {
    for(int i = 0; i < length - 1; i++) {
        if(array[i] > array[i + 1]) {
            printf("Array not sorted.\n");
            return;
        }
    }

    printf("Array sorted.\n");
}
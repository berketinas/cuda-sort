#include "includes.h"
#include "utils.h"
#include "bubbles.h"

int main(int argc, char* argv[]) {
    srand(time(NULL));

    int length = 1024;
    int* block = (int*) malloc(sizeof(int));
    int* grid = (int*) malloc(sizeof(int));

    *block = 256;
    *grid = 4;

    int* h_array = (int*) malloc(length * sizeof(int));
    
    // ALLOCATING SPACE FOR ARRAY ON THE GPU, AND OUTPUT ARRAY
    int* d_array;
    int* d_output = (int*) malloc(length * sizeof(int));
    cudaMalloc(&d_array, length * sizeof(int));

    // INITIALIZING THE HOST AND DEVICE ARRAYS
    init(h_array, length);
    cudaMemcpy(d_array, h_array, length * sizeof(int), cudaMemcpyHostToDevice);

    float time;
    cudaEvent_t start, stop;

    // PREPARATIONS TO TIME CPU EXECUTION
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // PERFORMING BUBBLE SORT ON CPU
    printf("Performing bubble sort on CPU. Array currently not sorted.\n");
    cpuBubble(h_array, length);
    
    // TIMING CPU EXECUTION
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("CPU execution duration: %f ms\n", time);

    printf("Checking CPU output: ");
    checkSorted(h_array, length);

    // PREPARATIONS TO TIME KERNEL EXECUTION
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // INVOKE BUBBLE KERNEL, ASSUME OUTPUT IS SORTED, AND INVOKE OUTPUT CHECK KERNEL
    // REPEAT UNTIL NONE OF THE THREADS IN OUTPUT CHECK KERNEL FIND ELEMENTS OUT OF ORDER
    gpuBubble(grid, block, d_array, length);
    handleError(cudaDeviceSynchronize());
    handleError(cudaGetLastError());

    // TIMING THE GPU EVENT
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("GPU execution duration: %f ms\n", time);

    // COPYING THE RESULT FROM THE DEVICE TO HOST
    cudaMemcpy(d_output, d_array, length * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Checking GPU output: ");
    checkSorted(d_output, length);

    return 0;
}
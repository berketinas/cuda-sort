#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// RANDOMLY INITIALIZE INPUT INTEGER ARRAY
void init(int* array, int length) {
    for(int i = 0; i < length; i++) {
        array[i] = rand();
    }
}

// PRINT OUTPUT ARRAY
void printOutput(int* array, int length) {
    printf("Output: [");
    for(int i = 0; i < length - 1; i++) printf("%d, ", array[i]);
    printf("%d]\n", array[length - 1]);
}

// CHECK WHETHER OUTPUT ARRAY WAS SORTED
void checkOutput(int* array, int length) {
    for(int i = 0; i < length - 1; i++) {
        if(array[i] > array[i + 1]) {
            printf("Array not sorted.\n");
            return;
        }
    }

    printf("Array sorted.\n");
}

// CLASSIC NON-PARALLEL BUBBLE SORT
void cpuBubble(int* array, int length) {
    int temp;
    for(int i = 0; i < length - 1; i++) {
        for(int j = 0; j < length - i - 1; j++) {
            if(array[j] > array[j + 1]) {
                temp = array[j];
                array[j] = array[j + 1];
                array[j + 1] = temp;
            }
        }
    }
}

// EACH THREAD COMPARES TWO ADJACENT ELEMENTS
__global__ void checkOutputGpu(int* array, int length, bool* sorted) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index < length - 1) {
        if(array[index] > array[index + 1]) {
            *sorted = false;
        }
    }
}

// EACH THREAD SWAPS TWO ADJACENT ELEMENTS
__global__ void gpuBubble(int* array, int length, int offset) {
    int index = 2 * (blockDim.x * blockIdx.x + threadIdx.x) + offset;

    if(index < length - (offset + 1)) {
        if(array[index] > array[index + 1]) {
            int temp = array[index];
            array[index] = array[index + 1];
            array[index + 1] = temp;
        }
    }
}

int main(int argc, char* argv[]) {
    srand(time(NULL));

    int length = 2048;
    int* array = (int*) malloc(length * sizeof(int));
    
    // ALLOCATING SPACE FOR ARRAY ON THE GPU, AND OUTPUT ARRAY
    int* arrayGpu;
    int* outputGpu = (int*) malloc(length * sizeof(int));
    cudaMalloc(&arrayGpu, length * sizeof(int));

    // ALLOCATING SPACE FOR THE SORTED FLAG
    bool* h_sorted = (bool*) malloc(sizeof(bool));
    bool* sorted;
    cudaMalloc(&sorted, sizeof(bool));

    // INITIALIZING THE HOST AND DEVICE ARRAYS
    init(array, length);
    cudaMemcpy(arrayGpu, array, length * sizeof(int), cudaMemcpyHostToDevice);

    float time;
    cudaEvent_t start, stop;

    // PREPARATIONS TO TIME CPU EXECUTION
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // PERFORMING BUBBLE SORT ON CPU
    printf("Performing bubble sort on CPU. Array currently not sorted.\n");
    cpuBubble(array, length);
    
    // TIMING CPU EXECUTION
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("CPU execution duration: %f ms\n", time);

    printf("Checking CPU output: ");
    checkOutput(array, length);

    // PREPARATIONS TO TIME KERNEL EXECUTION
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // LOOP COUNTER
    int iteration = 0;

    cudaEventRecord(start, 0);

    // KERNEL INVOCATION AND SYNC TO HALT CPU UNTIL GPU WORK IS FINISHED
    do {
        gpuBubble<<<1, 1024>>>(arrayGpu, length, iteration % 2);
        cudaDeviceSynchronize();

        iteration++;
        *h_sorted = true;
        cudaMemcpy(sorted, h_sorted, sizeof(bool), cudaMemcpyHostToDevice);

        checkOutputGpu<<<2, 1024>>>(arrayGpu, length, sorted);
        cudaDeviceSynchronize();

        cudaMemcpy(h_sorted, sorted, sizeof(bool), cudaMemcpyDeviceToHost);
    } while(!(*h_sorted));

    cudaDeviceSynchronize();

    // TIMING THE GPU EVENT
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("GPU execution duration: %f ms\n", time);

    // COPYING THE RESULT FROM THE DEVICE TO HOST
    cudaMemcpy(outputGpu, arrayGpu, length * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Checking GPU output: ");
    checkOutput(outputGpu, length);

    return 0;
}
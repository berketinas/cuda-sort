#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime_api.h> 
#include <cuda.h> 
#include <cooperative_groups.h>

// EXCEPTION HANDLING FUNCTION TO PRINT ERRORS, 
// SOURCED FROM THE INTERNET
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

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

// EACH THREAD REPEATEDLY SWAPS ADJACENT ELEMENTS
__global__ void gpuBubble(int* array, int length) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int temp;

    // CURRENTLY ONLY WORKS FOR SINGLE BLOCK CONFIGURATIONS, 
    // DO NOT FORGET TO REPLACE group.sync() WITH __syncthreads()

    cooperative_groups::grid_group group = cooperative_groups::this_grid(); 

    for(int i = 0; i < length / 2; i++) {
        if(!(index % 2) && index < length - 1) {
            if(array[index] > array[index+1]) {
                temp = array[index + 1];
                array[index + 1] = array[index];
                array[index] = temp;
            }
        }
        group.sync();
        // __syncthreads();

        if(index % 2 && index < length - 1) {
            if(array[index] > array[index+1]) {
                temp = array[index + 1];
                array[index + 1] = array[index];
                array[index] = temp;
            }
        }
        group.sync();
        // __syncthreads();
    }
}

int main(int argc, char* argv[]) {
    srand(time(NULL));

    int length = 1024;
    int threadX = 1024;
    int gridX = (length / threadX) + (length % threadX ? 1 : 0);

    int* array = (int*) malloc(length * sizeof(int));
    
    // ALLOCATING SPACE FOR ARRAY ON THE GPU, AND OUTPUT ARRAY
    int* arrayGpu;
    int* outputGpu = (int*) malloc(length * sizeof(int));
    cudaMalloc(&arrayGpu, length * sizeof(int));

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

    cudaEventRecord(start, 0);

    // INVOKE BUBBLE KERNEL, ASSUME OUTPUT IS SORTED, AND INVOKE OUTPUT CHECK KERNEL
    // REPEAT UNTIL NONE OF THE THREADS IN OUTPUT CHECK KERNEL FIND ELEMENTS OUT OF ORDER
    gpuBubble<<<gridX, threadX>>>(arrayGpu, length);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaGetLastError());

    // TIMING THE GPU EVENT
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("GPU execution duration: %f ms\n", time);

    // COPYING THE RESULT FROM THE DEVICE TO HOST
    cudaMemcpy(outputGpu, arrayGpu, length * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Checking GPU output: ");
    checkOutput(outputGpu, length);
    // printOutput(outputGpu, length);

    return 0;
}
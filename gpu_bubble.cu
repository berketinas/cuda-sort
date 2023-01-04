#include "includes.h"

// WARP
// ARRAY STORAGE IN C IS CONTIGUOUS IN MEMORY, COALESCING POSSIBLE
// SHARED MEMORY IF LIMITED TO ONE BLOCK

using namespace cooperative_groups;

__device__ void d_swap(int* x, int* y) {
    int temp = *x;
    *x = *y;
    *y = temp;
}

// EACH THREAD REPEATEDLY SWAPS ADJACENT ELEMENTS
__global__ void kernel(int* array, int length) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;    

    auto g = this_grid();

    for(int i = 0; i < length / 2; i++) {
        if(!(index % 2) && index < length - 1) {
            if(array[index] > array[index + 1]) {
                d_swap(&array[index], &array[index + 1]);
            }
        }
        g.sync();

        if(index % 2 && index < length - 1) {
            if(array[index] > array[index + 1]) {
                d_swap(&array[index], &array[index + 1]);
            }
        }
        g.sync();
    }
}

void gpuBubble(int* grid, int* block, int* array, int length) {
    int dev = 0;
    int supportsCoopLaunch = 0;

    cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);
    printf("supports coop: %d\n", supportsCoopLaunch);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("MAX_BLOCKS: %d\n", deviceProp.maxBlocksPerMultiProcessor);
    printf("MAX_THREADS_PER_BLOCK: %d\n", deviceProp.maxThreadsPerBlock);

    void* args[] = { &array, &length };

    cudaLaunchCooperativeKernel((void*) kernel, *grid, *block, args);
}
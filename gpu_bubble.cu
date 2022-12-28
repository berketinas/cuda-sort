// WARP
// ARRAY STORAGE IN C IS CONTIGUOUS IN MEMORY, COALESCING POSSIBLE
// SHARED MEMORY IF LIMITED TO ONE BLOCK

__device__ void d_swap(int* x, int* y) {
    int temp = *x;
    *x = *y;
    *y = temp;
}

// EACH THREAD REPEATEDLY SWAPS ADJACENT ELEMENTS
__global__ void kernel(int* array, int length) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    // CURRENTLY ONLY WORKS FOR SINGLE BLOCK CONFIGURATIONS, 
    // DO NOT FORGET TO REPLACE group.sync() WITH __syncthreads()

    // cooperative_groups::grid_group group = cooperative_groups::this_grid(); 

    for(int i = 0; i < length / 2; i++) {
        if(!(index % 2) && index < length - 1) {
            if(array[index] > array[index+1]) {
                d_swap(&array[index], &array[index + 1]);
            }
        }
        // group.sync();
        __syncthreads();

        if(index % 2 && index < length - 1) {
            if(array[index] > array[index+1]) {
                d_swap(&array[index], &array[index + 1]);
            }
        }
        // group.sync();
        __syncthreads();
    }
}

void gpuBubble(dim3 grid, dim3 block, int* array, int length) {
    kernel<<<grid, block>>>(array, length);
}
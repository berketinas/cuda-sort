#ifndef BUBBLES_H
#define BUBBLES_H

void cpuBubble(int* array, int length);
void gpuBubble(dim3 grid, dim3 block, int* array, int length);

#endif
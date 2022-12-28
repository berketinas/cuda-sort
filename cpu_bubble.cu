__host__ void h_swap(int* x, int* y) {
    int temp = *x;
    *x = *y;
    *y = temp;
}

void cpuBubble(int* array, int length) {
    for(int i = 0; i < length - 1; i++) {
        for(int j = 0; j < length - i - 1; j++) {
            if(array[j] > array[j + 1]) {
                h_swap(&array[j], &array[j + 1]);
            }
        }
    }
}
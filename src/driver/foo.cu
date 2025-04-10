#include <stdio.h>

// Minimal CUDA kernel that does nothing
__global__ void kernel() {}

int main() {
    printf("Launching minimal CUDA kernel\n");
    
    // Launch the kernel with 1 block and 1 thread
    kernel<<<1, 1>>>();
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();
    
    printf("Kernel execution completed\n");
    
    return 0;
}

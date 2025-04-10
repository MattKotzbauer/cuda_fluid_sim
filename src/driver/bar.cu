#include <stdio.h>

__global__ void DiffuseKernel(
    float* DiffusionTarget,
    const float* TargetPrior,
    float DiffusionConstant,
    int SimulationWidth,
    int SimulationHeight
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= 1 && i <= SimulationWidth && j >= 1 && j <= SimulationHeight)
    {
        int idx = i + (SimulationWidth + 2) * j;
        DiffusionTarget[idx] = (TargetPrior[idx] +
            DiffusionConstant * (
                DiffusionTarget[idx - 1] + 
                DiffusionTarget[idx + 1] +
                DiffusionTarget[idx - (SimulationWidth+2)] + 
                DiffusionTarget[idx + (SimulationWidth+2)]
            )
        ) / (1.f + 4.f * DiffusionConstant);
    }
}

int main() {
    // Simulation parameters
    int width = 128;
    int height = 128;
    float diffConst = 0.2f;
    
    // Allocate memory (including border cells)
    int totalSize = (width + 2) * (height + 2);
    float *d_target, *d_prior;
    cudaMalloc(&d_target, totalSize * sizeof(float));
    cudaMalloc(&d_prior, totalSize * sizeof(float));
    
    // Define grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, 
                 (height + blockDim.y - 1) / blockDim.y);
    
    // Call the kernel
    DiffuseKernel<<<gridDim, blockDim>>>(d_target, d_prior, diffConst, width, height);
    
    // Free memory
    cudaFree(d_target);
    cudaFree(d_prior);
    
    return 0;
}

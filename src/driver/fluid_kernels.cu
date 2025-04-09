#include <cuda_runtime.h>
#include "fluid_kernels.h"

// GPU kernel that performs one Gauss–Seidel sweep.
__global__ void DiffuseKernel(
    float* DiffusionTarget,
    const float* TargetPrior,
    float DiffusionConstant,
    int SimulationWidth,
    int SimulationHeight
)
{
    // Each thread corresponds to (i, j)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Convert (i, j) to 1D index. Note your original code used:
    //   IX(i, j) = i + (SimulationWidth + 2) * j
    // But you likely want to guard i,j in [1..SimulationWidth, 1..SimulationHeight].
    if (i >= 1 && i <= SimulationWidth && j >= 1 && j <= SimulationHeight)
    {
        int idx = i + (SimulationWidth + 2) * j;

        // One sweep of the Gauss–Seidel formula
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

// This is the "bridge" function you call from your C++ code instead of NSDiffuse().
// It runs multiple Gauss–Seidel iterations + boundary logic on the GPU.
extern "C"
void NSDiffuse_GPU(
    int Mode,
    float* d_DiffusionTarget,   // device pointer
    float* d_TargetPrior,       // device pointer
    float DiffusionConstant,
    int SimulationWidth,
    int SimulationHeight,
    int GaussSeidelIterations
)
{
    // 1. Decide block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (SimulationWidth + blockSize.x) / blockSize.x,
        (SimulationHeight + blockSize.y) / blockSize.y
    );

    // 2. Launch Gauss–Seidel sweeps (20 is typical, but user can pass in any number)
    for(int iter = 0; iter < GaussSeidelIterations; ++iter)
    {
        DiffuseKernel<<<gridSize, blockSize>>>(
            d_DiffusionTarget,
            d_TargetPrior,
            DiffusionConstant,
            SimulationWidth,
            SimulationHeight
        );
        
        // You might also call a GPU-based boundary kernel here, or do
        // cudaDeviceSynchronize() if you need to enforce strict iteration boundaries.
        // For example:
        // BoundKernel<<<...>>>(Mode, d_DiffusionTarget, SimulationWidth, SimulationHeight);
    }

    // 3. (Optional) final sync or boundary correction
    // cudaDeviceSynchronize();
}

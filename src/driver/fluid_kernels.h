#ifndef FLUID_KERNELS_H
#define FLUID_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

void NSDiffuse_GPU(
    int Mode,
    float* d_DiffusionTarget,
    float* d_TargetPrior,
    float DiffusionConstant,
    int SimulationWidth,
    int SimulationHeight,
    int GaussSeidelIterations
);

#ifdef __cplusplus
}
#endif

#endif // FLUID_KERNELS_H

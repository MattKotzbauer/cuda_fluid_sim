@echo off

:: Create the bin/ folder
mkdir ..\..\bin
pushd ..\..\bin

:: 1) Compile CUDA kernel(s) to .obj (or .o on some toolchains)
nvcc -c -o fluid_kernels.obj ..\src\driver\fluid_kernels.cu ^
     --compiler-options /std:c++17

:: 2) Compile your main CPU code with cl
cl -FC -Zi /std:c++17 ..\src\driver\driver.cpp ^
    user32.lib Gdi32.lib fluid_kernels.obj

popd

@echo off

:: Create the bin/ folder if it doesn't already exist.
mkdir ..\..\bin
pushd ..\..\bin

rem nvcc -arch=sm_86 -v -o bar.exe ..\src\driver\bar.cu

rem nvcc -arch=sm_86 -v -o foo.exe ..\src\driver\foo.cu

rem nvcc -arch=sm_86 -v -o fluid.exe ..\src\driver\fluid_kernels.cu

rem FUCKING WORKING
nvcc -arch=sm_86 -c ..\src\driver\fluid_kernels.cu -o fluid_kernels.obj

:: 1) Compile fluid_kernels.cu, targeting sm_86
rem nvcc -arch=sm_86 -v -c -o fluid_kernels.obj ..\src\driver\fluid_kernels.cu ^
     rem --compiler-options /std:c++17

:: 2) Compile driver.cu (formerly driver.cpp), also targeting sm_86
rem nvcc -arch=sm_86 -v -c -o driver.obj ..\src\driver\driver.cu ^
     rem --compiler-options /std:c++17

:: 3) Link all objects into an .exe, pulling in user32.lib and gdi32.lib
rem nvcc -arch=sm_86 -o fluid_sim.exe driver.obj fluid_kernels.obj user32.lib Gdi32.lib

popd

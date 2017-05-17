/* CUDA finite difference wave equation solver, written by
 * Jeff Amelang, 2012
 *
 * Modified by Kevin Yuh, 2013-14 */

#include <cstdio>

#include <cuda_runtime.h>

#include "Cuda1DFDWave_cuda.cuh"


/* TODO: You'll need a kernel here, as well as any helper functions
to call it */
void cudaDisplacementKernel(const float *old_d, const float *current_d, float *new_d, float size, float courant){
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;
  while(idx < numberOfNodes - 2){
    new_d[idx] = 2 * current_d[idx] - old_d[idx] + courant * courant * (current_d[idx + 1] - 2 * current_d[idx] + current_d[idx - 1]);
    idx += blockDim.x + gridDim.x;
  }
}

void cudaCallDisplacementKernel(const unsigned int blocks, const unsigned int threadsPerBlock, const float *old_d, const float *current_d, const float *new_d, float courant){
  cudaDisplacementKernel<<<blocks, threadsPerBlock>>>(old_d, current_d, new_d, courant);
}

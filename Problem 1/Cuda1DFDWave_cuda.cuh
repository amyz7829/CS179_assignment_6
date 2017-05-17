/* CUDA finite difference wave equation solver, written by
 * Jeff Amelang, 2012
 *
 * Modified by Kevin Yuh, 2013-14 */

#ifndef CUDA_1D_FD_WAVE_CUDA_CUH
#define CUDA_1D_FD_WAVE_CUDA_CUH


/* TODO: This is a CUDA header file.
If you have any functions in your .cu file that need to be
accessed from the outside, declare them here */
void cudaCallDisplacementKernel(const unsigned int blocks, const unsigned int threadsPerBlock,
  const float *old_d, const float *current_d, float *new_d, float size, float courant);

#endif // CUDA_1D_FD_WAVE_CUDA_CUH

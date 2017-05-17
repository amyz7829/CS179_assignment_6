#include <assert.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <curand.h>
#include "ta_utilities.hpp"

#define B         10.0
#define G         1.0
#define KON       0.1
#define KOFF      0.9

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// TODO: 2.1     Gillespie timestep implementation (25 pts)
__global__
void gillespieTimestepKernel()
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
}

// TODO: 2.2     Data resampling and stopping condition (25 pts)
__global__
void gillespieResampleKernel(int * completed)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
}

// TODO: 2.3a    Calculation of system mean (10 pts)
__global__
void gillespieAccumulateMeans()
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
}

// TODO: 2.3b    Calculation of system varience (10 pts)
__global__
void gillespieAccumulateVariances()
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        printf("usage: %s <threads per block> <number of blocks>\n",
                argv[0]);
        exit(EXIT_FAILURE);
    }

    const unsigned nThreads = atoi(argv[1]);
    const unsigned nBlocks  = atoi(argv[2]);

    // These functions allow you to select the least utilized GPU
    // on your system as well as enforce a time limit on program execution.
    // Please leave these enabled as a courtesy to your fellow classmates
    // if you are using a shared computer. You may ignore or remove these
    // functions if you are running on your local machine.
    TA_Utilities::select_least_utilized_GPU();
    int max_time_allowed_in_seconds = 30;
    TA_Utilities::enforce_time_limit(max_time_allowed_in_seconds);

    /* State variables for the Monte Carlo simulations. */
    int simComplete;
    int *d_simComplete; 

    /* Allocate memory on the GPU. */
    cudaMalloc((void **)&d_simComplete, sizeof(int));

    /* Perform initialization */

    /* Loop over each timestep in the simulation. */
    do {
        /* Generate random numbers for the simulation. */

        /* Execute a single timestep in the Gillespie simulation. */
        gillespieTimestepKernel<<<nBlocks, nThreads>>>();

        /* Haven't completed yet. */
        gpuErrchk( cudaMemset(d_simComplete, 0, sizeof(int)) );

        /* Accumulate the results of the timestep. */
        gillespieResampleKernel<<<nBlocks, nThreads>>>(d_simComplete);

        /* Check if stopping condition has been reached. */
        cudaMemcpy(&simComplete, d_simComplete, sizeof(int), cudaMemcpyDeviceToHost);

    } while (simComplete == false);

    /* Gather the results. */
    gillespieAccumulateMeans<<<nBlocks, nThreads, nThreads * sizeof(int)>>>();
    gillespieAccumulateVariances<<<nBlocks, nThreads, nThreads*sizeof(int)>>>();


    /* Free memory */
}

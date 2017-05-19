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

enum state : int{
  OFF = 0, ON = 1, DONE = 2
};

// TODO: 2.1     Gillespie timestep implementation (25 pts)
// Given an array of random numbers, want to determine propensities
__global__
void gillespieTimestepKernel(float *times, int *states, int *concentrations, float *rand_transitions, float *rand_timesteps, int size)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while(idx < size){
      //If state is not done
      if(states[idx] != 2){
        float timestep;
        float log_x = log(rand_timesteps[idx]);
        // State is currently OFF
        if(states[idx] == 0){
          // Off -> On
          float total_propensities = .1 + concentrations[idx] * G;
          if(rand_transitions[idx] < .1 / total_propensities){
            states[idx] = 1;
            timestep = log_x / total_propensities;
          }
          // [X] --
          else{
            timestep = log_x / (concentrations[idx] * G / total_propensities);
            if(concentrations[idx] == 0){
            }
            else{
              concentrations[idx] = concentrations[idx] - 1;
            }
          }
        }
        // State is currently ON
        else{
          float total_propensities = .9 + B + concentrations[idx] * G;
          // On -> Off
          if(rand_transitions[idx] < .9 / total_propensities){
            states[idx] = 0;
            timestep = log_x / total_propensities;
          }
          else if(rand_transitions[idx] < (.9 + 10) / total_propensities){
            concentrations[idx] = concentrations[idx] + 1;
            timestep = log_x / total_propensities;
          }
          else{
            concentrations[idx] = concentrations[idx] - 1;
            timestep = log_x / total_propensities;
          }
          times[idx] = times[idx] + timestep;
        }
      }
      idx += gridDim.x + blockDim.x;
    }
}

// TODO: 2.2     Data resampling and stopping condition (25 pts)
__global__
void gillespieResampleKernel(float *times, float *states, int *concentrations, float *d_timesteps, int *standard_concentrations, int * completed, int size)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while(idx < size){
      // As long as the state is not done
      if(states[idx] != 2){
        // Find the "starting time" for this timestep
        int i;
        for(i = (times[idx] - d_timesteps[idx]) / .1; i < times[idx] / .1; i++){
          standard_concentrations[i * 1000 + idx] = concentrations[idx];
        }
        if(times[idx] > 100){
          states[idx] = 2;
        }
        else{
          completed = false;
        }
      }
    }
}

// TODO: 2.3a    Calculation of system mean (10 pts)
__global__
void gillespieAccumulateMeans(int * standard_concentrations, float *means, int num_timesteps, int size)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int block_idx = blockIdx.x;
    extern __shared__ int data[];

    data[idx] = 0;
    while(block_idx < num_timesteps){
      while(idx < size){
        data[threadIdx.x] += standard_concentrations[block_idx * 1000 + idx];
        idx += gridDim.x * blockDim.x;
      }
      __syncthreads();
      if(threadIdx.x == 0){
        int sum = data[threadIdx.x];
        for(unsigned int i = 1; i < blockDim.x; i++){
          sum += data[i];
        }
        atomicAdd(&means[block_idx], (float)sum / num_timesteps);
        block_idx += gridDim.x;
      }
    }
}

// TODO: 2.3b    Calculation of system varience (10 pts)
__global__
void gillespieAccumulateVariances(int *standard_concentrations, float *variances, float *means, int num_timesteps, int size)
{
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int block_idx = blockIdx.x;
  extern __shared__ float data[];

  data[idx] = 0;
  while(block_idx < num_timesteps){
    while(idx < size){
      data[threadIdx.x] += powf(means[block_idx] - standard_concentrations[block_idx * 1000 + idx], 2);
      idx += gridDim.x * blockDim.x;
    }
    __syncthreads();
    if(threadIdx.x == 0){
      int sum = data[threadIdx.x];
      for(unsigned int i = 1; i < blockDim.x; i++){
        sum += data[i];
      }
      atomicAdd(&variances[block_idx], (float)sum / num_timesteps);
      block_idx += gridDim.x;
    }
  }
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

    float *dev_random_timesteps;
    float *dev_random_transitions;

    float *dev_times;
    int *dev_states;
    int *dev_concentrations;

    float *dev_d_timesteps;
    float *dev_uniform_samples;

    float *dev_means;
    float *dev_variances;

    /* Allocate memory on the GPU. */
    cudaMalloc((void **)&d_simComplete, sizeof(int));

    cudaMalloc((void **)&dev_random_timesteps, sizeof(float) * 4 * nThreads);
    cudaMalloc((void **)&dev_random_transitions, sizeof(float) * 4 * nThreads);

    cudaMalloc((void **)&dev_times, sizeof(float) * nThreads);
    cudaMalloc((void **)&dev_states, sizeof(int) * nThreads);
    cudaMalloc((void **)&dev_concentrations, sizeof(int) * 4  * nThreads);

    cudaMalloc((void **)&dev_d_timesteps, sizeof(float) * nThreads);
    // All 0th values are idx, 1st values are 1 * 1000 + idx, etc..
    cudaMalloc((void **)&dev_uniform_samples, sizeof(int) *  4 * nThreads * 1000);

    cudaMalloc((void **)&dev_means, sizeof(float) * 1000);
    cudaMalloc((void **)&dev_variances, sizeof(float) * 1000);

    cudaMemset(dev_times, 0, sizeof(float) * 4 * nThreads);
    cudaMemset(dev_states, 0, sizeof(int) * 4 * nThreads);
    cudaMemset(dev_concentrations, 0, sizeof(int) * 4 * nThreads);
    cudaMemset(dev_uniform_samples, 0, sizeof(int) *  4 * nThreads);

    /* Perform initialization */
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

    /* Loop over each timestep in the simulation. */
    do {
        /* Generate random numbers for the simulation. */
        cudaGenerateUniform(gen, dev_random_transitions, nThreads);
        cudaGenerateUniform(gen, dev_random_timesteps, nThreads);

        /* Execute a single timestep in the Gillespie simulation. */
        gillespieTimestepKernel<<<nBlocks, nThreads>>>(dev_times, dev_states, dev_concentrations, dev_random_transitions, dev_random_timesteps, 4 * nThreads);

        /* Haven't completed yet. */
        gpuErrchk( cudaMemset(d_simComplete, 0, sizeof(int)) );

        /* Accumulate the results of the timestep. */
        cudaMemset(d_simComplete, 1, sizeof(int));
        gillespieResampleKernel<<<nBlocks, nThreads>>>(dev_times, dev_states, dev_concentrations, dev_d_timesteps, dev_uniform_samples, d_simComplete, 4 * nThreads);

        /* Check if stopping condition has been reached. */
        cudaMemcpy(&simComplete, d_simComplete, sizeof(int), cudaMemcpyDeviceToHost);

    } while (simComplete == false);

    /* Gather the results. */
      gillespieAccumulateMeans<<<nBlocks, nThreads, nThreads * sizeof(int)>>>(dev_uniform_samples, dev_means, 1000, 4 * nThreads);
      gillespieAccumulateVariances<<<nBlocks, nThreads, nThreads * sizeof(float)>>>(dev_uniform_samples, dev_variances, dev_means, 1000, 4 * nThreads);


    /* Free memory */
}

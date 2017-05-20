#include <assert.h>
#include <stdio.h>
#include <iostream>
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
        if(states[idx] == OFF){
          // Off -> On
          float total_propensities = .1 + concentrations[idx] * G;
          if(rand_transitions[idx] < .1 / total_propensities){
            states[idx] = ON;
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
            states[idx] = OFF;
            timestep = log_x / total_propensities;
          }
          // [X]++
          else if(rand_transitions[idx] < (.9 + 10) / total_propensities){
            concentrations[idx] = concentrations[idx] + 1;
            timestep = log_x / total_propensities;
          }
          // [X]--
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

// TODO: 2.2 Data resampling and stopping condition (25 pts)
__global__
void gillespieResampleKernel(float *times, int *states, int *concentrations, float *d_timesteps, int *standard_concentrations, int * not_completed, int size)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while(idx < size){
      // As long as the state is not done
      if(states[idx] != 2){
        // Find the "starting time" for this timestep
        int i;
        // For all timesteps we have gone through since the last resampling, set those time steps to current concentration
        for(i = (times[idx] - d_timesteps[idx]) / .1; i < times[idx] / .1; i++){
          standard_concentrations[i * 1000 + idx] = concentrations[idx];
        }
        // If any simulations have passed the max time, set their state to DONE
        if(times[idx] > 100){
          states[idx] = DONE;
        }
        // Otherwise, if any simulations are still under time 100, then we are not completed, so we set not completed to true
        else{
          *not_completed = 1;
        }
      }
    }
}

// Reduction is used within a thread to accumulate the sum, and then the sum / num_timesteps is added to the total mean at that timestep
__global__
void gillespieAccumulateMeans(int * standard_concentrations, float *means, int num_timesteps, int size)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int block_idx = blockIdx.x;
    extern __shared__ int sums[];

    sums[idx] = 0;
    while(block_idx < num_timesteps){
      while(idx < size){
        sums[threadIdx.x] += standard_concentrations[block_idx * 1000 + idx];
        idx += gridDim.x * blockDim.x;
      }
      __syncthreads();
      if(threadIdx.x == 0){
        int sum = sums[threadIdx.x];
        for(unsigned int i = 1; i < blockDim.x; i++){
          sum += sums[i];
        }
        atomicAdd(&means[block_idx], (float)sum / num_timesteps);
        block_idx += gridDim.x;
      }
    }
}

// Reduction is used within a thread to accumulate the sum of variances, and that sum / num_timesteps is added to the total variance at that timestep
__global__
void gillespieAccumulateVariances(int *standard_concentrations, float *variances, float *means, int num_timesteps, int size)
{
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int block_idx = blockIdx.x;
  extern __shared__ float var_sums[];

  var_sums[idx] = 0;
  while(block_idx < num_timesteps){
    while(idx < size){
      var_sums[threadIdx.x] += powf(means[block_idx] - standard_concentrations[block_idx * 1000 + idx], 2);
      idx += gridDim.x * blockDim.x;
    }
    __syncthreads();
    if(threadIdx.x == 0){
      int sum = var_sums[threadIdx.x];
      for(unsigned int i = 1; i < blockDim.x; i++){
        sum += var_sums[i];
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

    int num_simulations = 4000;

    float *dev_random_timesteps;
    float *dev_random_transitions;

    float *dev_times;
    int *dev_states;
    int *dev_concentrations;

    float *dev_d_timesteps;
    int *dev_uniform_samples;

    float *dev_means;
    float *dev_variances;

    /* Allocate memory on the GPU. */
    cudaMalloc((void **)&d_simComplete, sizeof(int));

    cudaMalloc((void **)&dev_random_timesteps, sizeof(float) * num_simulations);
    cudaMalloc((void **)&dev_random_transitions, sizeof(float) * num_simulations);

    cudaMalloc((void **)&dev_times, sizeof(float) * num_simulations);
    cudaMalloc((void **)&dev_states, sizeof(int) * num_simulations);
    cudaMalloc((void **)&dev_concentrations, sizeof(int) * num_simulations);

    cudaMalloc((void **)&dev_d_timesteps, sizeof(float) * nThreads);
    // All 0th values are idx, 1st values are 1 * 1000 + idx, etc..
    cudaMalloc((void **)&dev_uniform_samples, sizeof(int) *  num_simulations * 1000);

    cudaMalloc((void **)&dev_means, sizeof(float) * 1000);
    cudaMalloc((void **)&dev_variances, sizeof(float) * 1000);

    cudaMemset(dev_times, 0, sizeof(float) * num_simulations);
    cudaMemset(dev_states, 0, sizeof(int) * num_simulations);
    cudaMemset(dev_concentrations, 0, sizeof(int) * num_simulations);
    cudaMemset(dev_uniform_samples, 0, sizeof(int) *  num_simulations);

    /* Perform initialization */
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

    /* Loop over each timestep in the simulation. */
    do {
        /* Generate random numbers for the simulation. */
        curandGenerateUniform(gen, dev_random_transitions, num_simulations);
        curandGenerateUniform(gen, dev_random_timesteps, num_simulations);

        /* Execute a single timestep in the Gillespie simulation. */
        gillespieTimestepKernel<<<nBlocks, nThreads>>>(dev_times, dev_states, dev_concentrations, dev_random_transitions, dev_random_timesteps, num_simulations);
        std::cout<<"completed timestep kernel"<<std::endl;

        //Set not_completed variable to "false", so we can detect if we still need to iterate
        gpuErrchk(cudaMemset(d_simComplete, 0, sizeof(int)));
        /* Accumulate the results of the timestep. */
        gillespieResampleKernel<<<nBlocks, nThreads>>>(dev_times, dev_states, dev_concentrations, dev_d_timesteps, dev_uniform_samples, d_simComplete, num_simulations);
        std::cout<<"completed resample kernel"<<std::endl;

        /* Check if stopping condition has been reached. */
        gpuErrchk(cudaMemcpy(&simComplete, d_simComplete, sizeof(int), cudaMemcpyDeviceToHost));

    } while (simComplete != 0);
    std::cout<<"Out of loop"<<std::endl;
    /* Gather the results. */
      gillespieAccumulateMeans<<<nBlocks, nThreads, nThreads * sizeof(int)>>>(dev_uniform_samples, dev_means, 1000, num_simulations);
      gillespieAccumulateVariances<<<nBlocks, nThreads, nThreads * sizeof(float)>>>(dev_uniform_samples, dev_variances, dev_means, 1000, num_simulations);

      float means[1000];
      float variances[1000];
      cudaMemcpy(means, dev_means, 1000 * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(variances, dev_variances, 1000 * sizeof(float), cudaMemcpyDeviceToHost);
      for(int i = 0; i < 1000; i++){
        std::cout<<"Mean: "<<means[i]<<std::endl;
        std::cout<<"Variance: "<<variances[i]<<std::endl;
      }

    /* Free memory */
    curandDestroyGenerator(gen);
    cudaFree(d_simComplete);
    cudaFree(dev_times);
    cudaFree(dev_states);
    cudaFree(dev_concentrations);
    cudaFree(dev_random_timesteps);
    cudaFree(dev_uniform_samples);
    cudaFree(dev_d_timesteps);
    cudaFree(dev_random_transitions);
    cudaFree(dev_means);
    cudaFree(dev_variances);
}

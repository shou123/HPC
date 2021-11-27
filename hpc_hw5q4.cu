#include <assert.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iomanip>
#include <numeric>
#include <random>
#include <stdio.h>
#include <vector>

#ifdef PARALLEL
#include <omp.h>
#endif

#include <../common/helper_cuda.h>
#include <../common/helper_functions.h>
#include <cuda_runtime.h>

constexpr auto NThread{1024};
constexpr auto NIter{1000000};

__global__ void setupRandomState(curandState *State) {
  auto Tid{threadIdx.x + blockDim.x * blockIdx.x};
  if (Tid > NThread)
    return;
  curand_init(1234, Tid, 0, &State[Tid]);
}

__device__ double calculatePi(int N, curandState *State, unsigned int Tid) {
  auto Count{0};
  for (auto I{0}; I < N; ++I) {
    auto X{curand_uniform(&State[Tid])};
    auto Y{curand_uniform(&State[Tid])};
    Count += (std::pow(X, 2) + std::pow(Y, 2) <= 1);
  }
  return Count * 1.0 / N * 4;
}

__global__ void calculatePiGpu(double *Sum, curandState *State) {
  auto Tid{blockIdx.x * blockDim.x + threadIdx.x};
  if (Tid > NThread)
    return;
  auto Pi{calculatePi(NIter, State, Tid)};
  atomicAdd(&Sum[0], Pi);
  //   Sum[Tid] = Pi;
}

int main(int argc, char **argv) {

  // This will pick the best possible CUDA capable device
  int DevId;
  DevId = findCudaDevice(argc, (const char **)argv);

  // Get GPU information
  cudaDeviceProp Props;
  checkCudaErrors(cudaGetDevice(&DevId));
  checkCudaErrors(cudaGetDeviceProperties(&Props, DevId));
  printf("Device %d: \"%s\" with Compute %d.%d capability\n", DevId, Props.name,
         Props.major, Props.minor);

  using T = double;
  auto TotSize{sizeof(T) * 1};

  int ThreadsPerBlock = 128;
  int BlocksPerGrid = (NThread + ThreadsPerBlock - 1) / ThreadsPerBlock;

  // Set random state
  curandState *State;
  cudaMalloc(&State, sizeof(curandState) * NThread);
  setupRandomState<<<BlocksPerGrid, ThreadsPerBlock>>>(State);

  T *HostSum = nullptr;
  T *DeviceSum = nullptr;
  HostSum = (T *)malloc(TotSize);
  memset(HostSum, static_cast<T>(0.0), TotSize);
  cudaMalloc(reinterpret_cast<void **>(&DeviceSum), TotSize);
  cudaMemset(DeviceSum, 0.0, TotSize);

  calculatePiGpu<<<BlocksPerGrid, ThreadsPerBlock>>>(DeviceSum, State);
  cudaMemcpy(HostSum, DeviceSum, TotSize, cudaMemcpyDeviceToHost);

  //   std::for_each(HostSum, HostSum + NThread,
  //                 [&](T I) { std::cout << I << " "; });
  std::cout << std::setprecision(32) << HostSum[0] / NThread << std::endl;

  return EXIT_SUCCESS;
}
